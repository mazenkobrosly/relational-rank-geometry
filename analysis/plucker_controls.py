#!/usr/bin/env python3
"""Shared utilities for Plucker sign-entropy robustness analyses.

The functions in this module are intentionally small and file-format light.
They are used by the runner scripts in this directory and can also be imported
from notebooks.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_LAYERS = [10, 20, 30, 40, 50, 55, 60, 70, 79]
DEFAULT_RANKS = [2, 3, 4]
DEFAULT_SEEDS = list(range(10))
DEFAULT_PROJ_DIM = 64
DEFAULT_MAX_LENGTH = 512
DEFAULT_N_HUB_TOKENS = 50
DEFAULT_N_TUPLES = 200
DEFAULT_N_CANDIDATES = 5000
DEFAULT_MIN_GAP = 2


@dataclass(frozen=True)
class TupleConfig:
    n_hub_tokens: int = DEFAULT_N_HUB_TOKENS
    n_selected_tuples: int = DEFAULT_N_TUPLES
    n_control_tuples: int = DEFAULT_N_TUPLES
    n_candidates: int = DEFAULT_N_CANDIDATES
    min_gap: int = DEFAULT_MIN_GAP
    position_bins: int = 10
    norm_bins: int = 10


def parse_int_list(raw: Optional[str], default: Sequence[int]) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return list(default)
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_float_list(raw: Optional[str], default: Sequence[float]) -> List[float]:
    if raw is None or str(raw).strip() == "":
        return list(default)
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def ensure_parent(path: Path) -> None:
    path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_parent(path)
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(str(key))
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values with monotone correction."""
    p = np.asarray([1.0 if x is None or not np.isfinite(x) else float(x) for x in p_values], dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    q = np.ones(n, dtype=float)
    for rank, idx in enumerate(order, start=1):
        q[idx] = p[idx] * n / rank
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0.0, 1.0)


def summarize_delta_group(rows: Sequence[Mapping], group_keys: Sequence[str], delta_key: str = "delta") -> pd.DataFrame:
    """Return mean/std/t-test rows grouped by keys."""
    if not rows:
        return pd.DataFrame(columns=list(group_keys) + ["n", "mean_delta", "std_delta", "sem_delta", "t_stat", "p_value", "q_value"])
    df = pd.DataFrame(rows)
    if df.empty or delta_key not in df.columns:
        return pd.DataFrame()

    out_rows = []
    for keys, g in df.groupby(list(group_keys), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(g[delta_key], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else float("nan")
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0
        if n > 1 and std > 0:
            t_stat, p_value = stats.ttest_1samp(vals, popmean=0.0, alternative="greater")
            t_stat = float(t_stat)
            p_value = float(p_value)
        else:
            t_stat = float("nan")
            p_value = 1.0
        row = {key: val for key, val in zip(group_keys, keys)}
        row.update({
            "n": n,
            "mean_delta": mean,
            "std_delta": std,
            "sem_delta": sem,
            "t_stat": t_stat,
            "p_value": p_value,
        })
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["q_value"] = bh_qvalues(out["p_value"].to_numpy(dtype=float))
    return out


def answer_token_candidates(tokenizer, letter: str) -> List[int]:
    candidates = set()
    patterns = [
        letter,
        " " + letter,
        "(" + letter,
        letter + ")",
        "(" + letter + ")",
        " " + letter + ")",
        "\n" + letter,
    ]
    for pattern in patterns:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if ids:
            candidates.add(int(ids[0]))
    return sorted(candidates)


def format_gpqa_prompt(item: Mapping, tokenizer) -> str:
    prompt = f"Question: {item['question']}\n\nChoices:\n"
    for i, choice in enumerate(item.get("choices", [])):
        prompt += f"  ({chr(65 + i)}) {choice}\n"
    prompt += "\nAnswer: The correct answer is ("
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _normalize_gpqa_item(row: Mapping) -> Dict:
    question = row.get("question") or row.get("Question") or row.get("prompt_body") or row.get("q") or ""
    choices = row.get("choices")
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except Exception:
            choices = None
    if not choices:
        option_keys = ["option_a", "option_b", "option_c", "option_d"]
        if all(key in row for key in option_keys):
            choices = [row[key] for key in option_keys]
    if not choices:
        answer_keys = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
        if all(key in row for key in answer_keys):
            choices = [row[key] for key in answer_keys]
    item = {"question": question, "choices": list(choices or [])}
    if "correct_idx" in row and row["correct_idx"] not in (None, ""):
        item["correct_idx"] = int(row["correct_idx"])
    elif "answer" in row and isinstance(row["answer"], int):
        item["correct_idx"] = int(row["answer"])
    else:
        correct_answer = row.get("correct_answer") or row.get("Correct Answer")
        if correct_answer and item["choices"]:
            answer_l = str(correct_answer).strip().lower()
            if answer_l in {"a", "b", "c", "d"}:
                item["correct_idx"] = ord(answer_l) - ord("a")
            else:
                lowered = [str(c).strip().lower() for c in item["choices"]]
                if answer_l in lowered:
                    item["correct_idx"] = lowered.index(answer_l)
    qid = row.get("qid", row.get("q_idx"))
    if qid not in (None, ""):
        item["qid"] = int(qid)
    return item


def load_gpqa_items(path: Optional[str] = None) -> List[Dict]:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    root = Path(__file__).resolve().parents[1]
    candidates.extend([
        root / "gpqa_experiment" / "outputs" / "gpqa_items.json",
        root / "gpqa_experiment" / "outputs" / "gpqa_items.csv",
        Path("/ANON_LOCAL_PATH/Downloads/gpqa_experiment_405b_bundle_20260301_151302/gpqa_198_mcq.json"),
        Path("/ANON_LOCAL_PATH/Downloads/gpqa_diamond.csv"),
        Path("/ANON_LOCAL_PATH/annealing/dataset/gpqa_diamond.csv"),
    ])
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".csv":
            with candidate.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = json.loads(candidate.read_text(encoding="utf-8"))
        return [_normalize_gpqa_item(row) for row in rows]
    raise FileNotFoundError("Could not find GPQA items. Pass --gpqa-items explicitly.")


def attach_missing_correct_idx(items: List[Dict]) -> List[Dict]:
    if all("correct_idx" in item for item in items):
        return items
    correct_by_question: Dict[str, str] = {}
    for path in [
        Path("/ANON_LOCAL_PATH/Downloads/gpqa_diamond.csv"),
        Path("/ANON_LOCAL_PATH/annealing/dataset/gpqa_diamond.csv"),
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = row.get("Question") or row.get("question")
                ans = row.get("Correct Answer") or row.get("correct_answer")
                if q and ans:
                    correct_by_question[q.strip().lower()] = ans
        break
    for i, item in enumerate(items):
        if "correct_idx" in item:
            continue
        ans = correct_by_question.get(str(item.get("question", "")).strip().lower())
        if not ans:
            raise RuntimeError(f"Could not recover correct answer for item {i}")
        lowered = [str(c).strip().lower() for c in item.get("choices", [])]
        ans_l = ans.strip().lower()
        if ans_l in lowered:
            item["correct_idx"] = lowered.index(ans_l)
            continue
        import difflib
        sims = [difflib.SequenceMatcher(None, ans_l, choice).ratio() for choice in lowered]
        best = int(np.argmax(sims))
        if sims[best] < 0.75:
            raise RuntimeError(f"Could not align correct answer to choices for item {i}")
        item["correct_idx"] = best
    return items


def deterministic_projection(hidden_dim: int, proj_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    return rng.randn(int(hidden_dim), int(proj_dim)).astype(np.float32) / math.sqrt(float(proj_dim))


def bidirectional_attention(attn_matrix: np.ndarray) -> np.ndarray:
    """Average heads and symmetrize a [heads, tokens, tokens] attention tensor."""
    arr = np.asarray(attn_matrix, dtype=np.float64)
    if arr.ndim == 2:
        attn_avg = arr
    else:
        attn_avg = arr.mean(axis=0)
    return (attn_avg + attn_avg.T) / 2.0


def min_gap_ok(tup: Sequence[int], min_gap: int = DEFAULT_MIN_GAP) -> bool:
    ordered = sorted(int(x) for x in tup)
    return all(ordered[i + 1] - ordered[i] >= int(min_gap) for i in range(len(ordered) - 1))


def select_hub_tokens(attn_matrix: np.ndarray, n_hub: int = DEFAULT_N_HUB_TOKENS, min_gap: int = DEFAULT_MIN_GAP) -> List[int]:
    bidir = bidirectional_attention(attn_matrix)
    scores = bidir.sum(axis=1)
    selected: List[int] = []
    for idx in np.argsort(-scores):
        idx = int(idx)
        if all(abs(idx - prev) >= min_gap for prev in selected):
            selected.append(idx)
        if len(selected) >= n_hub:
            break
    return sorted(selected)


def pairwise_hub_attention(attn_matrix: np.ndarray, hubs: Sequence[int]) -> np.ndarray:
    bidir = bidirectional_attention(attn_matrix)
    idx = np.asarray(list(hubs), dtype=int)
    return bidir[np.ix_(idx, idx)]


def tuple_relation_score(hub_attn: np.ndarray, hub_tuple: Sequence[int]) -> float:
    if len(hub_tuple) < 2:
        return 0.0
    vals = [hub_attn[i, j] for a, i in enumerate(hub_tuple) for j in hub_tuple[a + 1:]]
    return float(np.min(vals)) if vals else 0.0


def find_high_attention_tuples(
    hub_attn: np.ndarray,
    hub_tokens: Sequence[int],
    k: int,
    n_tuples: int = DEFAULT_N_TUPLES,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    seed: int = 42,
    min_gap: int = DEFAULT_MIN_GAP,
) -> List[Tuple[int, ...]]:
    n_hub = len(hub_tokens)
    if n_hub < k:
        return []
    rng = np.random.RandomState(int(seed))
    scored: List[Tuple[float, Tuple[int, ...]]] = []
    for _ in range(int(n_candidates)):
        hi = tuple(sorted(int(x) for x in rng.choice(n_hub, size=int(k), replace=False)))
        tokens = tuple(int(hub_tokens[h]) for h in hi)
        if not min_gap_ok(tokens, min_gap):
            continue
        scored.append((tuple_relation_score(hub_attn, hi), tokens))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Tuple[int, ...]] = []
    seen = set()
    for _, tup in scored:
        if tup in seen:
            continue
        seen.add(tup)
        out.append(tup)
        if len(out) >= int(n_tuples):
            break
    return out


def find_random_tuples(
    seq_len: int,
    k: int,
    n_tuples: int = DEFAULT_N_TUPLES,
    seed: int = 99,
    min_gap: int = DEFAULT_MIN_GAP,
    exclude: Optional[Iterable[Tuple[int, ...]]] = None,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    forbidden = set(exclude or [])
    out: List[Tuple[int, ...]] = []
    seen = set()
    max_attempts = max(int(n_tuples) * 200, 500)
    for _ in range(max_attempts):
        tup = tuple(sorted(int(x) for x in rng.choice(int(seq_len), size=int(k), replace=False)))
        if tup in seen or tup in forbidden or not min_gap_ok(tup, min_gap):
            continue
        seen.add(tup)
        out.append(tup)
        if len(out) >= int(n_tuples):
            break
    return out


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=int)
    ranks = stats.rankdata(values, method="average") - 1.0
    denom = max(1.0, float(values.size - 1))
    bins = np.floor((ranks / denom) * int(n_bins)).astype(int)
    return np.clip(bins, 0, int(n_bins) - 1)


def find_position_matched_tuples(
    selected_tuples: Sequence[Tuple[int, ...]],
    seq_len: int,
    seed: int,
    n_bins: int = 10,
    min_gap: int = DEFAULT_MIN_GAP,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    bins = np.floor(np.arange(int(seq_len)) * int(n_bins) / max(1, int(seq_len))).astype(int)
    bins = np.clip(bins, 0, int(n_bins) - 1)
    by_bin = {b: np.where(bins == b)[0].astype(int).tolist() for b in range(int(n_bins))}
    out: List[Tuple[int, ...]] = []
    seen = set()
    for source in selected_tuples:
        pattern = [int(bins[min(max(int(t), 0), int(seq_len) - 1)]) for t in source]
        for _ in range(200):
            picked = []
            for b in pattern:
                pool = by_bin.get(b) or list(range(int(seq_len)))
                picked.append(int(rng.choice(pool)))
            tup = tuple(sorted(picked))
            if len(set(tup)) == len(tup) and min_gap_ok(tup, min_gap) and tup not in seen:
                seen.add(tup)
                out.append(tup)
                break
        else:
            fallback = find_random_tuples(seq_len, len(source), 1, seed=int(rng.randint(0, 2**31 - 1)), min_gap=min_gap, exclude=seen)
            if fallback:
                seen.add(fallback[0])
                out.append(fallback[0])
    return out


def find_norm_matched_tuples(
    selected_tuples: Sequence[Tuple[int, ...]],
    norms: np.ndarray,
    seed: int,
    n_bins: int = 10,
    min_gap: int = DEFAULT_MIN_GAP,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    norms = np.asarray(norms, dtype=float)
    bins = _quantile_bins(norms, n_bins)
    by_bin = {b: np.where(bins == b)[0].astype(int).tolist() for b in range(int(n_bins))}
    out: List[Tuple[int, ...]] = []
    seen = set()
    for source in selected_tuples:
        pattern = [int(bins[min(max(int(t), 0), len(bins) - 1)]) for t in source]
        for _ in range(300):
            picked = []
            for b in pattern:
                pool = by_bin.get(b) or list(range(len(norms)))
                picked.append(int(rng.choice(pool)))
            tup = tuple(sorted(picked))
            if len(set(tup)) == len(tup) and min_gap_ok(tup, min_gap) and tup not in seen:
                seen.add(tup)
                out.append(tup)
                break
        else:
            fallback = find_random_tuples(len(norms), len(source), 1, seed=int(rng.randint(0, 2**31 - 1)), min_gap=min_gap, exclude=seen)
            if fallback:
                seen.add(fallback[0])
                out.append(fallback[0])
    return out


def classify_token_text(token: str) -> str:
    raw = str(token)
    stripped = raw.strip()
    if stripped == "" or raw in {"<s>", "</s>", "<pad>"} or raw.startswith("<|"):
        return "whitespace_or_special"
    clean = stripped.lstrip("Ġ▁").strip()
    if clean in {"A", "B", "C", "D", "(A)", "(B)", "(C)", "(D)", "A)", "B)", "C)", "D)"}:
        return "answer_label"
    if clean and all(ch in string.punctuation for ch in clean):
        return "punctuation"
    if re.fullmatch(r"[-+]?(\d+([.,]\d+)*|\d*\.\d+)([%a-zA-Z]*)", clean or ""):
        return "number"
    if re.search(r"[A-Za-z]", clean or ""):
        return "alphabetic"
    return "other"


def token_classes_from_ids(tokenizer, input_ids: Sequence[int]) -> List[str]:
    try:
        tokens = tokenizer.convert_ids_to_tokens([int(x) for x in input_ids])
    except Exception:
        tokens = [str(int(x)) for x in input_ids]
    return [classify_token_text(tok) for tok in tokens]


def find_token_class_matched_tuples(
    selected_tuples: Sequence[Tuple[int, ...]],
    token_classes: Sequence[str],
    seed: int,
    min_gap: int = DEFAULT_MIN_GAP,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    by_class: Dict[str, List[int]] = {}
    for i, cls in enumerate(token_classes):
        by_class.setdefault(str(cls), []).append(int(i))
    out: List[Tuple[int, ...]] = []
    seen = set()
    for source in selected_tuples:
        pattern = [token_classes[int(t)] for t in source]
        for _ in range(400):
            picked = []
            for cls in pattern:
                pool = by_class.get(str(cls)) or list(range(len(token_classes)))
                picked.append(int(rng.choice(pool)))
            tup = tuple(sorted(picked))
            if len(set(tup)) == len(tup) and min_gap_ok(tup, min_gap) and tup not in seen:
                seen.add(tup)
                out.append(tup)
                break
        else:
            fallback = find_random_tuples(len(token_classes), len(source), 1, seed=int(rng.randint(0, 2**31 - 1)), min_gap=min_gap, exclude=seen)
            if fallback:
                seen.add(fallback[0])
                out.append(fallback[0])
    return out


def find_degree_decoy_tuples(
    hub_attn: np.ndarray,
    hub_tokens: Sequence[int],
    true_tuples: Sequence[Tuple[int, ...]],
    k: int,
    n_tuples: int,
    seed: int,
    min_gap: int = DEFAULT_MIN_GAP,
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> List[Tuple[int, ...]]:
    """High individual attention degree, low mutual tuple relation."""
    n_hub = len(hub_tokens)
    if n_hub < k:
        return []
    rng = np.random.RandomState(int(seed))
    degree = np.asarray(hub_attn, dtype=float).sum(axis=1)
    top_count = max(int(k) + 2, min(n_hub, max(12, int(math.ceil(0.7 * n_hub)))))
    top_hub = np.argsort(-degree)[:top_count]
    token_to_hub = {int(tok): i for i, tok in enumerate(hub_tokens)}
    true_degree_means = []
    for tup in true_tuples:
        hubs = [token_to_hub.get(int(t)) for t in tup]
        if all(h is not None for h in hubs):
            true_degree_means.append(float(np.mean(degree[hubs])))
    min_degree_mean = float(np.percentile(true_degree_means, 25)) if true_degree_means else float(np.percentile(degree[top_hub], 25))

    scored: List[Tuple[float, float, Tuple[int, ...]]] = []
    for _ in range(int(n_candidates)):
        hi = tuple(sorted(int(x) for x in rng.choice(top_hub, size=int(k), replace=False)))
        tokens = tuple(sorted(int(hub_tokens[h]) for h in hi))
        if not min_gap_ok(tokens, min_gap):
            continue
        deg_mean = float(np.mean(degree[list(hi)]))
        if deg_mean < min_degree_mean:
            continue
        rel = tuple_relation_score(hub_attn, hi)
        scored.append((rel, -deg_mean, tokens))
    scored.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, ...]] = []
    seen = set(true_tuples)
    for _, _, tup in scored:
        if tup in seen:
            continue
        seen.add(tup)
        out.append(tup)
        if len(out) >= int(n_tuples):
            break
    return out


def compute_basis(x_proj: np.ndarray, k: int, center: bool = False) -> np.ndarray:
    x = np.asarray(x_proj, dtype=np.float64)
    if center:
        x = x - np.mean(x, axis=0, keepdims=True)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, : int(k)]


def compute_minors(basis: np.ndarray, tuples: Sequence[Tuple[int, ...]]) -> List[float]:
    basis = np.asarray(basis, dtype=np.float64)
    minors: List[float] = []
    for tup in tuples:
        if len(tup) == 0:
            continue
        if all(0 <= int(t) < basis.shape[0] for t in tup):
            sub = basis[list(map(int, tup)), :]
            if sub.shape[0] == sub.shape[1]:
                sign, logabs = np.linalg.slogdet(sub)
                if sign == 0:
                    minors.append(0.0)
                else:
                    minors.append(float(sign * np.exp(logabs)))
    return minors


def sign_entropy(minors: Sequence[float], eps: float = 0.0) -> float:
    vals = np.asarray([float(m) for m in minors if np.isfinite(m) and abs(float(m)) > float(eps)], dtype=float)
    if vals.size < 2:
        return 0.0
    p = float(np.mean(vals > 0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def entropy_from_tuples(basis: np.ndarray, tuples: Sequence[Tuple[int, ...]], eps: float = 0.0) -> Tuple[float, List[float]]:
    minors = compute_minors(basis, tuples)
    return sign_entropy(minors, eps=eps), minors


def entropy_delta(
    basis: np.ndarray,
    selected_tuples: Sequence[Tuple[int, ...]],
    control_tuples: Sequence[Tuple[int, ...]],
    eps: float = 0.0,
) -> Dict[str, object]:
    selected_entropy, selected_minors = entropy_from_tuples(basis, selected_tuples, eps=eps)
    control_entropy, control_minors = entropy_from_tuples(basis, control_tuples, eps=eps)
    return {
        "H_selected": float(selected_entropy),
        "H_control": float(control_entropy),
        "delta": float(control_entropy - selected_entropy),
        "n_selected": len(selected_minors),
        "n_control": len(control_minors),
        "selected_mean_abs_det": float(np.mean(np.abs(selected_minors))) if selected_minors else float("nan"),
        "control_mean_abs_det": float(np.mean(np.abs(control_minors))) if control_minors else float("nan"),
        "selected_minors": selected_minors,
        "control_minors": control_minors,
    }


def margin_filtered_entropy_delta(
    selected_minors: Sequence[float],
    control_minors: Sequence[float],
    threshold_quantile: float,
) -> Dict[str, float]:
    selected = np.asarray(selected_minors, dtype=float)
    control = np.asarray(control_minors, dtype=float)
    combined = np.concatenate([np.abs(selected[np.isfinite(selected)]), np.abs(control[np.isfinite(control)])])
    if combined.size == 0:
        eps = float("inf")
    else:
        eps = float(np.quantile(combined, float(threshold_quantile)))
    selected_keep = selected[np.abs(selected) >= eps]
    control_keep = control[np.abs(control) >= eps]
    selected_entropy = sign_entropy(selected_keep)
    control_entropy = sign_entropy(control_keep)
    return {
        "threshold_quantile": float(threshold_quantile),
        "det_threshold": eps,
        "H_selected": float(selected_entropy),
        "H_control": float(control_entropy),
        "delta": float(control_entropy - selected_entropy),
        "n_selected": int(selected_keep.size),
        "n_control": int(control_keep.size),
    }


def answer_scores_from_logits(logits, answer_token_ids: Mapping[str, Sequence[int]]) -> Dict[str, float]:
    last = logits[0, -1, :]
    scores = {}
    for letter, tids in answer_token_ids.items():
        vals = []
        for tid in tids:
            if int(tid) < last.shape[0]:
                vals.append(float(last[int(tid)].detach().float().cpu().item()))
        scores[letter] = max(vals) if vals else float("-inf")
    return scores


def answer_margin(scores: Mapping[str, float], correct_idx: Optional[int]) -> Dict[str, object]:
    letters = ["A", "B", "C", "D"]
    pred_letter = max(scores, key=lambda k: scores[k])
    pred_idx = letters.index(pred_letter)
    sorted_vals = sorted((float(v), k) for k, v in scores.items() if np.isfinite(v))
    chosen_margin = float(sorted_vals[-1][0] - sorted_vals[-2][0]) if len(sorted_vals) >= 2 else float("nan")
    correct_margin = float("nan")
    is_correct = None
    correct_letter = None
    if correct_idx is not None and 0 <= int(correct_idx) < len(letters):
        correct_letter = letters[int(correct_idx)]
        others = [float(v) for k, v in scores.items() if k != correct_letter and np.isfinite(v)]
        correct_margin = float(scores[correct_letter] - max(others)) if others else float("nan")
        is_correct = pred_idx == int(correct_idx)
    return {
        "pred_letter": pred_letter,
        "pred_idx": pred_idx,
        "correct_letter": correct_letter,
        "is_correct": is_correct,
        "chosen_margin": chosen_margin,
        "correct_margin": correct_margin,
    }


def load_hf_model_and_tokenizer(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    device_map: str = "auto",
    load_in_4bit: bool = True,
    attn_implementation: str = "eager",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _dtype_from_env(env_name: str, default: str):
        raw = os.getenv(env_name, default).strip().lower()
        aliases = {
            "fp16": "float16",
            "float16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp32": "float32",
            "float32": "float32",
        }
        name = aliases.get(raw, raw)
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype for {env_name}: {raw!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=bool(local_files_only),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = _dtype_from_env("MODEL_TORCH_DTYPE", "float16")
    requested_device_map = str(device_map or "auto").strip().lower()
    load_on_mps = requested_device_map in {"mps", "mps:0"}
    kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
        "cache_dir": cache_dir,
        "local_files_only": bool(local_files_only),
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if not load_on_mps:
        kwargs["device_map"] = device_map
    if load_in_4bit and load_on_mps:
        print("[warn] 4-bit load requested on MPS; loading torch_dtype weights instead.", file=sys.stderr)
        load_in_4bit = False
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            compute_dtype = _dtype_from_env("BNB_4BIT_COMPUTE_DTYPE", "float16")
            quant_type = os.getenv("BNB_4BIT_QUANT_TYPE", "fp4").strip().lower()
            use_double_quant = os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "0").strip().lower() in {"1", "true", "yes", "on"}
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=use_double_quant,
            )
        except Exception as exc:
            print(f"[warn] 4-bit load requested but BitsAndBytesConfig is unavailable ({exc}); falling back to torch_dtype only.", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if load_on_mps:
        model = model.to("mps")
    model.eval()
    return tokenizer, model


def choose_question_indices(n_items: int, quick: bool = False, subset_size: Optional[int] = None, subset_seed: int = 123) -> List[int]:
    if quick:
        subset_size = 30 if subset_size is None else int(subset_size)
        return [int(x) for x in np.linspace(0, n_items - 1, min(subset_size, n_items), dtype=int)]
    if subset_size is None or int(subset_size) >= n_items:
        return list(range(n_items))
    rng = np.random.RandomState(int(subset_seed))
    return sorted(int(x) for x in rng.choice(n_items, size=int(subset_size), replace=False))


def iter_model_records(
    *,
    model_id: str,
    gpqa_items_path: Optional[str],
    layers: Sequence[int],
    max_length: int,
    quick: bool,
    subset_size: Optional[int],
    subset_seed: int,
    cache_dir: Optional[str],
    local_files_only: bool,
    device_map: str,
    load_in_4bit: bool,
    live_every: int,
    callback: Callable[[Dict], Sequence[Mapping]],
    selected_indices: Optional[Sequence[int]] = None,
    model_load_path: Optional[str] = None,
) -> List[Mapping]:
    """Run one model forward per prompt and call callback with hidden/attention payload.

    The callback receives CPU numpy arrays for hidden states and attention for
    requested layers only, plus tokenizer/input metadata.
    """
    import torch

    items = attach_missing_correct_idx(load_gpqa_items(gpqa_items_path))
    if selected_indices is not None:
        indices = [int(x) for x in selected_indices]
    else:
        indices = choose_question_indices(len(items), quick=quick, subset_size=subset_size, subset_seed=subset_seed)
    tokenizer, model = load_hf_model_and_tokenizer(
        model_id=model_load_path or model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )
    answer_ids = {letter: answer_token_candidates(tokenizer, letter) for letter in ["A", "B", "C", "D"]}
    rows: List[Mapping] = []
    t0 = time.time()
    for pos, qi in enumerate(indices, start=1):
        item = items[qi]
        prompt = format_gpqa_prompt(item, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length))
        input_ids = inputs["input_ids"][0].detach().cpu().numpy().astype(int)
        model_inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            out = model(**model_inputs, output_hidden_states=True, output_attentions=True)
        scores = answer_scores_from_logits(out.logits, answer_ids)
        behavior = answer_margin(scores, item.get("correct_idx"))
        hidden_by_layer = {}
        attn_by_layer = {}
        for layer in layers:
            if int(layer) + 1 < len(out.hidden_states):
                hidden_by_layer[int(layer)] = out.hidden_states[int(layer) + 1][0].detach().float().cpu().numpy()
            if int(layer) < len(out.attentions):
                attn_by_layer[int(layer)] = out.attentions[int(layer)][0].detach().float().cpu().numpy()
        payload = {
            "q_idx": int(qi),
            "item": item,
            "seq_len": int(inputs["input_ids"].shape[1]),
            "input_ids": input_ids,
            "token_classes": token_classes_from_ids(tokenizer, input_ids),
            "tokenizer": tokenizer,
            "hidden_by_layer": hidden_by_layer,
            "attn_by_layer": attn_by_layer,
            "answer_scores": scores,
            **behavior,
        }
        rows.extend(callback(payload))
        del out, model_inputs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if live_every and (pos % max(1, int(live_every)) == 0):
            elapsed = time.time() - t0
            print(f"[live] {pos}/{len(indices)} prompts done in {elapsed:.1f}s; latest q_idx={qi}", flush=True)
    return rows


def read_prompt_id_file(path: str | Path) -> List[int]:
    p = Path(path).expanduser()
    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(int(line))
    return ids


def iter_cached_payloads(cache_dir: str | Path, layers: Optional[Sequence[int]] = None, prompt_ids: Optional[Sequence[int]] = None) -> Iterable[Dict]:
    """Yield payloads compatible with runner callbacks from a saved NPZ cache.

    Cache layout:
      manifest.json
      prompts/prompt_0000.json
      arrays/prompt_0000.npz with hidden_L{layer}, bidir_L{layer}
    """
    root = Path(cache_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    available = {int(x) for x in (manifest.get("prompt_ids") or [])}
    if not available:
        available = {int(p.stem.split("_")[-1]) for p in (root / "arrays").glob("prompt_*.npz")}
    wanted = set(int(x) for x in prompt_ids) if prompt_ids is not None else available
    layer_filter = set(int(x) for x in layers) if layers is not None else None
    for q_idx in sorted(available & wanted):
        meta_path = root / "prompts" / f"prompt_{q_idx:04d}.json"
        arr_path = root / "arrays" / f"prompt_{q_idx:04d}.npz"
        if not arr_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        arrays = np.load(arr_path, allow_pickle=False)
        hidden_by_layer = {}
        attn_by_layer = {}
        for key in arrays.files:
            if key.startswith("hidden_L"):
                layer = int(key.replace("hidden_L", ""))
                if layer_filter is None or layer in layer_filter:
                    hidden_by_layer[layer] = arrays[key].astype(np.float32, copy=False)
            elif key.startswith("bidir_L"):
                layer = int(key.replace("bidir_L", ""))
                if layer_filter is None or layer in layer_filter:
                    attn_by_layer[layer] = arrays[key].astype(np.float32, copy=False)
        input_ids = arrays["input_ids"].astype(int) if "input_ids" in arrays.files else np.asarray(meta.get("input_ids", []), dtype=int)
        yield {
            "q_idx": int(q_idx),
            "item": meta.get("item") or {},
            "seq_len": int(meta.get("seq_len") or (len(input_ids) if input_ids is not None else 0)),
            "input_ids": input_ids,
            "token_classes": meta.get("token_classes") or [],
            "token_strings": meta.get("token_strings") or [],
            "span_metadata": meta.get("span_metadata") or {},
            "hidden_by_layer": hidden_by_layer,
            "attn_by_layer": attn_by_layer,
            "answer_scores": meta.get("answer_scores") or {},
            "pred_letter": meta.get("pred_letter"),
            "pred_idx": meta.get("pred_idx"),
            "correct_letter": meta.get("correct_letter"),
            "is_correct": meta.get("is_correct"),
            "chosen_margin": meta.get("chosen_margin"),
            "correct_margin": meta.get("correct_margin"),
        }
