#!/usr/bin/env python3
"""Phase 2 semantic-rank and k-wise selector audit.

This runner keeps the Phase 1 interpretation intact: high-degree selection is
reported as salience / semantic-hub selection, never as relation-free. It uses
the cached 70B GPQA hidden/attention arrays for true cross-selector tests, and
it separately flattens the existing relational ladder task-bank JSONs when the
task bank has metrics but no raw hidden/attention tensors.
"""

from __future__ import annotations

import argparse
import os
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.plucker_controls import (  # noqa: E402
    DEFAULT_PROJ_DIM,
    TupleConfig,
    bh_qvalues,
    compute_basis,
    deterministic_projection,
    entropy_from_tuples,
    iter_cached_payloads,
    parse_int_list,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "could", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "might", "must", "not",
    "of", "on", "one", "or", "our", "should", "so", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "two", "was", "we", "were", "what", "when", "which", "while", "with",
    "would", "you", "your",
}

SCAFFOLD_TERMS = {
    "system", "user", "assistant", "question", "choices", "answer", "correct",
    "following", "options", "option", "date", "today", "knowledge", "cutting",
}

RELATION_WORDS = {
    "because", "therefore", "implies", "imply", "cause", "causes", "caused",
    "relation", "related", "similar", "analogous", "analogy", "if", "then",
    "unless", "except", "whereas", "while", "depends", "corresponds", "matches",
    "proportional", "increases", "decreases", "greater", "less", "opposite",
}

SELECTOR_LABELS = {
    "individual_degree_salience": "individual attention degree / salience",
    "content_only_individual_degree_salience": "content-only individual degree / semantic hubs",
    "mutual_attention_min_pair": "minimum pairwise mutual attention",
    "mutual_attention_avg_pair": "average pairwise mutual attention",
    "triangle_closure_k3": "triangle closure for k=3",
    "common_neighbor_two_hop": "common-neighbor / two-hop attention relation",
    "diffusion_profile_similarity": "diffusion-row similarity",
    "hidden_state_cosine_cohesion": "hidden-state cosine cohesion",
    "graph_community_comembership": "graph community co-membership",
    "random_control": "random control",
    "position_matched_control": "position-matched control",
    "norm_matched_control": "norm-matched control",
    "content_only_random_control": "content-only random control",
    "original_task_bank_mutual_attention": "task-bank original mutual-attention selector",
}

REQUESTED_TASK_SELECTORS = [
    "mutual_attention_min_pair",
    "individual_degree_salience",
    "content_only_individual_degree_salience",
    "diffusion_profile_similarity",
    "common_neighbor_two_hop",
    "hidden_state_cosine_cohesion",
    "random_control",
    "position_matched_control",
    "norm_matched_control",
]


@dataclass
class PromptTokenMeta:
    clean: List[str]
    lower: List[str]
    classes: List[str]
    is_special: np.ndarray
    is_punct_or_format: np.ndarray
    is_number: np.ndarray
    is_answer_label: np.ndarray
    is_option_text: np.ndarray
    is_stopword: np.ndarray
    is_content_word: np.ndarray
    is_content_only_eligible: np.ndarray
    section: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_cache = ROOT / "remote_compute_70b_outputs" / "important_full_20260506" / "results" / "cache_70b_gpqa_controls_full198"
    p.add_argument("--input-cache-dir", default=str(default_cache))
    p.add_argument("--task-bank-dir", default=str(ROOT / "ouro_local" / "relational_k_ladder_fullrun"))
    p.add_argument("--layers", default="50,55,60")
    p.add_argument("--ranks", default="2,3,4")
    p.add_argument("--projection-seed", type=int, default=42)
    p.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    p.add_argument("--n-hub-tokens", type=int, default=50)
    p.add_argument("--n-tuples", type=int, default=200)
    p.add_argument("--n-candidates", type=int, default=3000)
    p.add_argument("--min-gap", type=int, default=2)
    p.add_argument("--position-bins", type=int, default=10)
    p.add_argument("--norm-bins", type=int, default=10)
    p.add_argument("--graph-top-neighbors", type=int, default=24)
    p.add_argument("--max-triplets-per-selector", type=int, default=120)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    return p.parse_args()


def clean_token(tok: object) -> str:
    raw = str(tok)
    if raw.startswith("<|") or raw in {"<s>", "</s>", "<pad>"}:
        return ""
    return raw.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").strip()


def wordish(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9+\-^]", "", str(text)).lower()


def infer_sections(clean: Sequence[str]) -> List[str]:
    section = ["scaffold"] * len(clean)
    mode = "scaffold"
    current_option: Optional[str] = None
    for i, tok in enumerate(clean):
        low = wordish(tok)
        if low == "question":
            mode = "question"
            current_option = None
        elif low == "choices":
            mode = "choices"
            current_option = None
        elif low == "answer":
            mode = "answer"
            current_option = None
        if mode == "choices" and low in {"a", "b", "c", "d"}:
            current_option = low.upper()
        section[i] = f"option_{current_option}" if mode == "choices" and current_option else mode
    return section


def token_meta(payload: Mapping) -> PromptTokenMeta:
    token_strings = payload.get("token_strings") or [str(x) for x in payload.get("input_ids", [])]
    classes = list(payload.get("token_classes") or ["other"] * len(token_strings))
    clean = [clean_token(t) for t in token_strings]
    lower = [wordish(t) for t in clean]
    section = infer_sections(clean)
    is_special = np.asarray([(not c) or cls == "whitespace_or_special" for c, cls in zip(clean, classes)], dtype=bool)
    is_answer_label = np.asarray([cls == "answer_label" for cls in classes], dtype=bool)
    is_number = np.asarray([cls == "number" for cls in classes], dtype=bool)
    is_punct = np.asarray([
        cls == "punctuation" or (bool(c) and not re.search(r"[A-Za-z0-9]", c))
        for c, cls in zip(clean, classes)
    ], dtype=bool)
    is_scaffold = np.asarray([l in SCAFFOLD_TERMS or sec in {"scaffold", "answer"} for l, sec in zip(lower, section)], dtype=bool)
    is_punct_or_format = is_punct | is_scaffold | is_special
    is_option_text = np.asarray([sec.startswith("option_") for sec in section], dtype=bool) & ~is_answer_label & ~is_punct & ~is_special
    is_stopword = np.asarray([l in STOPWORDS for l in lower], dtype=bool)
    alphabetic = np.asarray([cls == "alphabetic" and bool(re.search(r"[A-Za-z]", c)) for c, cls in zip(clean, classes)], dtype=bool)
    is_content_word = alphabetic & ~is_stopword & ~is_punct_or_format & ~is_answer_label
    is_content_only_eligible = is_content_word & ~is_number
    return PromptTokenMeta(
        clean=clean,
        lower=lower,
        classes=classes,
        is_special=is_special,
        is_punct_or_format=is_punct_or_format,
        is_number=is_number,
        is_answer_label=is_answer_label,
        is_option_text=is_option_text,
        is_stopword=is_stopword,
        is_content_word=is_content_word,
        is_content_only_eligible=is_content_only_eligible,
        section=section,
    )


def bidir_no_self(attn: np.ndarray) -> np.ndarray:
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=0)
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 0.0)
    return a


def min_gap_ok(tup: Sequence[int], min_gap: int) -> bool:
    ordered = sorted(int(x) for x in tup)
    return all(ordered[i + 1] - ordered[i] >= int(min_gap) for i in range(len(ordered) - 1))


def select_by_degree(a: np.ndarray, eligible: Optional[np.ndarray], n_hub: int, min_gap: int) -> List[int]:
    scores = np.asarray(a, dtype=float).sum(axis=1)
    selected: List[int] = []
    for idx in np.argsort(-scores):
        idx = int(idx)
        if eligible is not None and (idx >= len(eligible) or not bool(eligible[idx])):
            continue
        if all(abs(idx - prev) >= min_gap for prev in selected):
            selected.append(idx)
        if len(selected) >= n_hub:
            break
    return sorted(selected)


def random_tuples_from_pool(
    pool: Sequence[int],
    k: int,
    n_tuples: int,
    seed: int,
    min_gap: int,
    exclude: Optional[Iterable[Tuple[int, ...]]] = None,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    pool_arr = np.asarray(sorted(set(int(x) for x in pool)), dtype=int)
    if len(pool_arr) < k:
        return []
    forbidden = set(exclude or [])
    out: List[Tuple[int, ...]] = []
    seen = set()
    for _ in range(max(1000, n_tuples * 300)):
        tup = tuple(sorted(int(x) for x in rng.choice(pool_arr, size=k, replace=False)))
        if tup in seen or tup in forbidden or not min_gap_ok(tup, min_gap):
            continue
        seen.add(tup)
        out.append(tup)
        if len(out) >= n_tuples:
            break
    return out


def candidate_tuples_scored(
    pool: Sequence[int],
    k: int,
    n_tuples: int,
    n_candidates: int,
    seed: int,
    min_gap: int,
    score_fn: Callable[[Tuple[int, ...]], float],
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    pool_arr = np.asarray(sorted(set(int(x) for x in pool)), dtype=int)
    if len(pool_arr) < k:
        return []
    scored: List[Tuple[float, Tuple[int, ...]]] = []
    seen = set()
    for _ in range(max(int(n_candidates), int(n_tuples) * 50)):
        tup = tuple(sorted(int(x) for x in rng.choice(pool_arr, size=k, replace=False)))
        if tup in seen or not min_gap_ok(tup, min_gap):
            continue
        seen.add(tup)
        scored.append((float(score_fn(tup)), tup))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:n_tuples]]


def candidate_tuple_array(
    pool: Sequence[int],
    k: int,
    n_tuples: int,
    n_candidates: int,
    seed: int,
    min_gap: int,
) -> np.ndarray:
    """Generate the same kind of candidate tuple universe, but score later in bulk."""
    rng = np.random.RandomState(int(seed))
    pool_arr = np.asarray(sorted(set(int(x) for x in pool)), dtype=np.int32)
    if len(pool_arr) < k:
        return np.empty((0, int(k)), dtype=np.int32)
    out: List[Tuple[int, ...]] = []
    seen = set()
    for _ in range(max(int(n_candidates), int(n_tuples) * 50)):
        tup = tuple(sorted(int(x) for x in rng.choice(pool_arr, size=k, replace=False)))
        if tup in seen or not min_gap_ok(tup, min_gap):
            continue
        seen.add(tup)
        out.append(tup)
    if not out:
        return np.empty((0, int(k)), dtype=np.int32)
    return np.asarray(out, dtype=np.int32)


def top_scored_tuples(cands: np.ndarray, scores: np.ndarray, n_tuples: int) -> List[Tuple[int, ...]]:
    if cands.size == 0:
        return []
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-scores, kind="mergesort")[: int(n_tuples)]
    return [tuple(int(x) for x in cands[int(i)].tolist()) for i in order]


def pair_matrix_values(mat: np.ndarray, cands: np.ndarray) -> np.ndarray:
    if cands.size == 0 or cands.shape[1] < 2:
        return np.empty((0, 0), dtype=np.float32)
    vals = [mat[cands[:, i], cands[:, j]] for i, j in itertools.combinations(range(cands.shape[1]), 2)]
    return np.stack(vals, axis=1).astype(np.float32, copy=False)


def scored_tuples_from_array(
    cands: np.ndarray,
    n_tuples: int,
    score_fn: Callable[[np.ndarray], np.ndarray],
) -> List[Tuple[int, ...]]:
    if cands.size == 0:
        return []
    return top_scored_tuples(cands, score_fn(cands), n_tuples)


def sparse_weight_maps(neighbors: Sequence[np.ndarray], weights: Sequence[np.ndarray]) -> List[Dict[int, float]]:
    return [
        {int(n): float(w) for n, w in zip(ns, ws)}
        for ns, ws in zip(neighbors, weights)
    ]


def sparse_dot_maps(weight_maps: Sequence[Mapping[int, float]], i: int, j: int) -> float:
    wi = weight_maps[int(i)]
    wj = weight_maps[int(j)]
    if len(wi) > len(wj):
        wi, wj = wj, wi
    return float(sum(float(w) * float(wj.get(int(n), 0.0)) for n, w in wi.items()))


def two_hop_direct_maps(weight_maps: Sequence[Mapping[int, float]], i: int, j: int) -> float:
    total = 0.0
    target = int(j)
    for mid, w1 in weight_maps[int(i)].items():
        total += float(w1) * float(weight_maps[int(mid)].get(target, 0.0))
    return float(total)


def fill_pair_matrix(
    seq_len: int,
    subset: Sequence[int],
    pair_fn: Callable[[int, int], float],
    dtype=np.float32,
) -> np.ndarray:
    mat = np.zeros((int(seq_len), int(seq_len)), dtype=dtype)
    ids = sorted(set(int(x) for x in subset if 0 <= int(x) < int(seq_len)))
    for i in ids:
        for j in ids:
            if i == j:
                continue
            mat[i, j] = pair_fn(i, j)
    return mat


def position_matched_from_pool(
    selected: Sequence[Tuple[int, ...]],
    pool: Sequence[int],
    seq_len: int,
    seed: int,
    n_bins: int,
    min_gap: int,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    pool = sorted(set(int(x) for x in pool))
    bins = np.floor(np.arange(int(seq_len)) * int(n_bins) / max(1, int(seq_len))).astype(int)
    bins = np.clip(bins, 0, int(n_bins) - 1)
    by_bin: Dict[int, List[int]] = {b: [] for b in range(n_bins)}
    for idx in pool:
        by_bin[int(bins[idx])].append(idx)
    out: List[Tuple[int, ...]] = []
    seen = set()
    for source in selected:
        pattern = [int(bins[int(t)]) for t in source]
        for _ in range(500):
            picked = [int(rng.choice(by_bin.get(b) or pool)) for b in pattern]
            tup = tuple(sorted(picked))
            if len(set(tup)) == len(tup) and tup not in seen and min_gap_ok(tup, min_gap):
                seen.add(tup)
                out.append(tup)
                break
    return out


def norm_matched_from_pool(
    selected: Sequence[Tuple[int, ...]],
    pool: Sequence[int],
    norms: np.ndarray,
    seed: int,
    n_bins: int,
    min_gap: int,
) -> List[Tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    pool = sorted(set(int(x) for x in pool))
    ranks = stats.rankdata(np.asarray(norms, dtype=float), method="average") - 1.0
    bins = np.floor((ranks / max(1.0, len(norms) - 1.0)) * int(n_bins)).astype(int)
    bins = np.clip(bins, 0, int(n_bins) - 1)
    by_bin: Dict[int, List[int]] = {b: [] for b in range(n_bins)}
    for idx in pool:
        by_bin[int(bins[idx])].append(idx)
    out: List[Tuple[int, ...]] = []
    seen = set()
    for source in selected:
        pattern = [int(bins[int(t)]) for t in source]
        for _ in range(600):
            picked = [int(rng.choice(by_bin.get(b) or pool)) for b in pattern]
            tup = tuple(sorted(picked))
            if len(set(tup)) == len(tup) and tup not in seen and min_gap_ok(tup, min_gap):
                seen.add(tup)
                out.append(tup)
                break
    return out


def top_neighbor_graph(attn: np.ndarray, top_n: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[set]]:
    a = bidir_no_self(attn)
    n = a.shape[0]
    neighbors: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    sets: List[set] = []
    for i in range(n):
        row = a[i].copy()
        if top_n < n - 1:
            idx = np.argpartition(-row, top_n)[:top_n]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.argsort(-row)
        vals = row[idx].astype(float)
        keep = vals > 0
        idx = idx[keep].astype(int)
        vals = vals[keep]
        denom = float(vals.sum())
        vals = vals / denom if denom > 0 else vals
        neighbors.append(idx)
        weights.append(vals)
        sets.append(set(int(x) for x in idx))
    return neighbors, weights, sets


def weighted_dot_sparse(i: int, j: int, neighbors: Sequence[np.ndarray], weights: Sequence[np.ndarray]) -> float:
    wi = {int(n): float(w) for n, w in zip(neighbors[i], weights[i])}
    return float(sum(wi.get(int(n), 0.0) * float(w) for n, w in zip(neighbors[j], weights[j])))


def diffusion_pair(i: int, j: int, neighbors: Sequence[np.ndarray], weights: Sequence[np.ndarray]) -> float:
    denom = math.sqrt(weighted_dot_sparse(i, i, neighbors, weights) * weighted_dot_sparse(j, j, neighbors, weights))
    return float(weighted_dot_sparse(i, j, neighbors, weights) / denom) if denom > 0 else 0.0


def two_hop_pair(i: int, j: int, neighbors: Sequence[np.ndarray], weights: Sequence[np.ndarray]) -> float:
    total = 0.0
    for mid, w1 in zip(neighbors[i], weights[i]):
        mid = int(mid)
        for dest, w2 in zip(neighbors[mid], weights[mid]):
            if int(dest) == int(j):
                total += float(w1) * float(w2)
                break
    return float(total)


def pair_values(tup: Tuple[int, ...]) -> List[Tuple[int, int]]:
    return [(int(i), int(j)) for i, j in itertools.combinations(tup, 2)]


def graph_communities(neighbor_sets: Sequence[set], min_size: int = 3) -> Dict[int, int]:
    n = len(neighbor_sets)
    labels = list(range(n))
    for _ in range(8):
        changed = False
        for i in range(n):
            votes = Counter(labels[j] for j in neighbor_sets[i] if 0 <= int(j) < n)
            if not votes:
                continue
            new = votes.most_common(1)[0][0]
            if labels[i] != new:
                labels[i] = new
                changed = True
        if not changed:
            break
    counts = Counter(labels)
    return {i: (labels[i] if counts[labels[i]] >= min_size else -1) for i in range(n)}


def mean_pair_score(tup: Tuple[int, ...], score_pair: Callable[[int, int], float]) -> float:
    vals = [float(score_pair(i, j)) for i, j in pair_values(tup)]
    return float(np.mean(vals)) if vals else 0.0


def build_selector_tuples(
    a: np.ndarray,
    x_proj: np.ndarray,
    meta: PromptTokenMeta,
    neighbors: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    neighbor_sets: Sequence[set],
    communities: Mapping[int, int],
    k: int,
    tuple_cfg: TupleConfig,
    norms: np.ndarray,
    seed_base: int,
) -> Dict[str, List[Tuple[int, ...]]]:
    seq_len = int(a.shape[0])
    degree = a.sum(axis=1)
    all_pool = list(range(seq_len))
    content_pool = [int(i) for i, ok in enumerate(meta.is_content_only_eligible[:seq_len]) if bool(ok)]
    raw_hubs = select_by_degree(a, None, tuple_cfg.n_hub_tokens, tuple_cfg.min_gap)
    content_hubs = select_by_degree(a, meta.is_content_only_eligible, tuple_cfg.n_hub_tokens, tuple_cfg.min_gap)
    xp = np.asarray(x_proj, dtype=np.float32)
    xp = xp / np.maximum(np.linalg.norm(xp, axis=1, keepdims=True), 1e-12)

    relation_pool = content_hubs or raw_hubs
    matrix_pool = sorted(set(raw_hubs) | set(content_hubs) | set(relation_pool))
    weight_maps = sparse_weight_maps(neighbors, weights)

    common_mat = fill_pair_matrix(
        seq_len,
        matrix_pool,
        lambda i, j: (len(neighbor_sets[i] & neighbor_sets[j]) / len(neighbor_sets[i] | neighbor_sets[j]))
        if (neighbor_sets[i] | neighbor_sets[j]) else 0.0,
    )
    self_sparse = {int(i): max(sparse_dot_maps(weight_maps, i, i), 1e-12) for i in matrix_pool}
    diffusion_mat = fill_pair_matrix(
        seq_len,
        matrix_pool,
        lambda i, j: sparse_dot_maps(weight_maps, i, j) / math.sqrt(self_sparse.get(int(i), 1e-12) * self_sparse.get(int(j), 1e-12)),
    )
    twohop_common_mat = fill_pair_matrix(
        seq_len,
        matrix_pool,
        lambda i, j: common_mat[i, j] + 0.5 * (two_hop_direct_maps(weight_maps, i, j) + two_hop_direct_maps(weight_maps, j, i)),
    )
    hidden_mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    if matrix_pool:
        ids = np.asarray(matrix_pool, dtype=np.int32)
        hidden_mat[np.ix_(ids, ids)] = xp[ids] @ xp[ids].T
    community_mat = fill_pair_matrix(
        seq_len,
        matrix_pool,
        lambda i, j: (1.0 if communities.get(int(i), -1) != -1 and communities.get(int(i), -1) == communities.get(int(j), -1) else 0.0)
        + 0.1 * common_mat[i, j],
    )

    def candidate(pool: Sequence[int], seed: int) -> np.ndarray:
        return candidate_tuple_array(pool, k, tuple_cfg.n_selected_tuples, tuple_cfg.n_candidates, seed, tuple_cfg.min_gap)

    def degree_scores(cands: np.ndarray) -> np.ndarray:
        return degree[cands].mean(axis=1)

    def avg_matrix_scores(mat: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda cands: pair_matrix_values(mat, cands).mean(axis=1)

    def min_matrix_scores(mat: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda cands: pair_matrix_values(mat, cands).min(axis=1)

    def triangle_scores(cands: np.ndarray) -> np.ndarray:
        vals = np.maximum(pair_matrix_values(a, cands), 1e-12)
        return np.exp(np.log(vals).mean(axis=1))

    selectors = {
        "individual_degree_salience": scored_tuples_from_array(candidate(raw_hubs, 10100 + seed_base), tuple_cfg.n_selected_tuples, degree_scores),
        "content_only_individual_degree_salience": scored_tuples_from_array(candidate(content_hubs, 10200 + seed_base), tuple_cfg.n_selected_tuples, degree_scores),
        "mutual_attention_min_pair": scored_tuples_from_array(candidate(raw_hubs, 10300 + seed_base), tuple_cfg.n_selected_tuples, min_matrix_scores(a)),
        "mutual_attention_avg_pair": scored_tuples_from_array(candidate(raw_hubs, 10400 + seed_base), tuple_cfg.n_selected_tuples, avg_matrix_scores(a)),
        "triangle_closure_k3": scored_tuples_from_array(candidate(raw_hubs, 10500 + seed_base), tuple_cfg.n_selected_tuples, triangle_scores) if k == 3 else [],
        "common_neighbor_two_hop": scored_tuples_from_array(candidate(relation_pool, 10600 + seed_base), tuple_cfg.n_selected_tuples, avg_matrix_scores(twohop_common_mat)),
        "diffusion_profile_similarity": scored_tuples_from_array(candidate(relation_pool, 10700 + seed_base), tuple_cfg.n_selected_tuples, avg_matrix_scores(diffusion_mat)),
        "hidden_state_cosine_cohesion": scored_tuples_from_array(candidate(relation_pool, 10800 + seed_base), tuple_cfg.n_selected_tuples, avg_matrix_scores(hidden_mat)),
        "graph_community_comembership": scored_tuples_from_array(candidate(relation_pool, 10900 + seed_base), tuple_cfg.n_selected_tuples, avg_matrix_scores(community_mat)),
    }
    base = selectors["mutual_attention_avg_pair"] or selectors["mutual_attention_min_pair"] or selectors["individual_degree_salience"]
    selectors["random_control"] = random_tuples_from_pool(all_pool, k, tuple_cfg.n_control_tuples, 11000 + seed_base, tuple_cfg.min_gap, exclude=base)
    selectors["position_matched_control"] = position_matched_from_pool(base, all_pool, seq_len, 11100 + seed_base, tuple_cfg.position_bins, tuple_cfg.min_gap)
    selectors["norm_matched_control"] = norm_matched_from_pool(base, all_pool, norms, 11200 + seed_base, tuple_cfg.norm_bins, tuple_cfg.min_gap)
    selectors["content_only_random_control"] = random_tuples_from_pool(content_pool, k, tuple_cfg.n_control_tuples, 11300 + seed_base, tuple_cfg.min_gap, exclude=selectors["content_only_individual_degree_salience"])
    return selectors


def entropy_row(
    selector: str,
    tuples: Sequence[Tuple[int, ...]],
    basis: np.ndarray,
    baselines: Mapping[str, float],
) -> Dict[str, object]:
    h, minors = entropy_from_tuples(basis, tuples)
    row: Dict[str, object] = {
        "selector": selector,
        "selector_label": SELECTOR_LABELS.get(selector, selector),
        "available": bool(len(tuples)),
        "sign_entropy": float(h) if len(tuples) else np.nan,
        "n_tuples": int(len(tuples)),
        "n_minors": int(len(minors)),
        "mean_abs_det": float(np.mean(np.abs(minors))) if minors else np.nan,
    }
    for name, val in baselines.items():
        row[f"selector_minus_{name}_entropy"] = float(h - val) if len(tuples) and np.isfinite(val) else np.nan
        row[f"{name}_minus_selector_entropy"] = float(val - h) if len(tuples) and np.isfinite(val) else np.nan
    return row


def triplet_feature_rows(
    selector: str,
    tuples: Sequence[Tuple[int, ...]],
    a: np.ndarray,
    basis: np.ndarray,
    x_proj: np.ndarray,
    meta: PromptTokenMeta,
    neighbors: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    neighbor_sets: Sequence[set],
    communities: Mapping[int, int],
    max_rows: int = 120,
) -> List[Dict[str, object]]:
    xp = np.asarray(x_proj, dtype=np.float32)
    xp = xp / np.maximum(np.linalg.norm(xp, axis=1, keepdims=True), 1e-12)
    rows: List[Dict[str, object]] = []
    for tup_id, tup in enumerate(list(tuples)[:max_rows]):
        if len(tup) != 3 or not all(0 <= int(i) < basis.shape[0] for i in tup):
            continue
        pairs = pair_values(tup)
        p_attn = [float(a[i, j]) for i, j in pairs]
        common_j = []
        for i, j in pairs:
            inter = neighbor_sets[i] & neighbor_sets[j]
            union = neighbor_sets[i] | neighbor_sets[j]
            common_j.append(len(inter) / len(union) if union else 0.0)
        all_inter = set.intersection(*(neighbor_sets[int(i)] for i in tup)) if tup else set()
        all_union = set.union(*(neighbor_sets[int(i)] for i in tup)) if tup else set()
        sections = {meta.section[int(i)] for i in tup if int(i) < len(meta.section)}
        has_option = any(s.startswith("option_") for s in sections)
        has_question = "question" in sections
        has_relation_word = any(meta.lower[int(i)] in RELATION_WORDS for i in tup if int(i) < len(meta.lower))
        comms = [communities.get(int(i), -1) for i in tup]
        sub = basis[list(map(int, tup)), :]
        sign, logabs = np.linalg.slogdet(sub)
        minor = 0.0 if sign == 0 else float(sign * np.exp(logabs))
        rows.append({
            "selector": selector,
            "selector_label": SELECTOR_LABELS.get(selector, selector),
            "tuple_id": int(tup_id),
            "token_i": int(tup[0]),
            "token_j": int(tup[1]),
            "token_l": int(tup[2]),
            "tokens": " | ".join(meta.clean[int(i)] if int(i) < len(meta.clean) else "" for i in tup),
            "sections": " | ".join(meta.section[int(i)] if int(i) < len(meta.section) else "" for i in tup),
            "triangle_closure": float(math.exp(np.mean(np.log(np.maximum(p_attn, 1e-12))))),
            "average_pairwise_attention": float(np.mean(p_attn)),
            "minimum_pairwise_attention": float(np.min(p_attn)),
            "common_neighbor_overlap_all3": float(len(all_inter) / len(all_union)) if all_union else 0.0,
            "common_neighbor_overlap_pair_mean": float(np.mean(common_j)) if common_j else 0.0,
            "two_hop_connectivity": mean_pair_score(tup, lambda i, j: 0.5 * (two_hop_pair(i, j, neighbors, weights) + two_hop_pair(j, i, neighbors, weights))),
            "diffusion_row_similarity": mean_pair_score(tup, lambda i, j: diffusion_pair(i, j, neighbors, weights)),
            "graph_community_comembership": bool(len(set(comms)) == 1 and comms[0] != -1),
            "hidden_state_cosine_cohesion": float(np.mean([float(xp[i] @ xp[j]) for i, j in pairs])),
            "spans_question_and_answer_option": bool(has_question and has_option),
            "has_relation_phrase_token": bool(has_relation_word),
            "content_only": bool(all(int(i) < len(meta.is_content_only_eligible) and meta.is_content_only_eligible[int(i)] for i in tup)),
            "minor": minor,
            "minor_sign": int(np.sign(minor)),
            "abs_minor": float(abs(minor)),
        })
    return rows


def cache_prompt_ids(cache_dir: Path) -> List[int]:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = manifest.get("prompt_ids") or []
        if ids:
            return [int(x) for x in ids]
    return sorted(int(p.stem.split("_")[-1]) for p in (cache_dir / "arrays").glob("prompt_*.npz"))


def process_prompt_job(job: Tuple) -> Tuple[int, List[Dict[str, object]], List[Dict[str, object]]]:
    (
        cache_dir_raw,
        q_idx,
        layers,
        ranks,
        tuple_cfg,
        projection_seed,
        proj_dim,
        graph_top_neighbors,
        max_triplets_per_selector,
    ) = job
    cache_dir = Path(cache_dir_raw)
    payload = next(iter_cached_payloads(cache_dir, layers=layers, prompt_ids=[int(q_idx)]))
    meta = token_meta(payload)
    selector_rows: List[Dict[str, object]] = []
    triplet_rows: List[Dict[str, object]] = []
    for layer in layers:
        if layer not in payload["hidden_by_layer"] or layer not in payload["attn_by_layer"]:
            continue
        hidden = payload["hidden_by_layer"][layer].astype(np.float32, copy=False)
        a = bidir_no_self(payload["attn_by_layer"][layer])
        proj = deterministic_projection(hidden.shape[1], proj_dim, projection_seed)
        x_proj = hidden @ proj
        norms = np.sqrt(np.sum(x_proj * x_proj, axis=1))
        neighbors, weights, neighbor_sets = top_neighbor_graph(a, graph_top_neighbors)
        communities = graph_communities(neighbor_sets)
        for k in ranks:
            seed_base = int(q_idx) * 1000 + int(layer) * 20 + int(k)
            selectors = build_selector_tuples(a, x_proj, meta, neighbors, weights, neighbor_sets, communities, k, tuple_cfg, norms, seed_base)
            basis = compute_basis(x_proj, k, center=False)
            random_h, _ = entropy_from_tuples(basis, selectors.get("random_control", []))
            mutual_h, _ = entropy_from_tuples(basis, selectors.get("mutual_attention_avg_pair", []))
            content_degree_h, _ = entropy_from_tuples(basis, selectors.get("content_only_individual_degree_salience", []))
            baselines = {"random_control": random_h, "mutual_attention_avg_pair": mutual_h, "content_degree_salience": content_degree_h}
            for selector, tuples in selectors.items():
                row = entropy_row(selector, tuples, basis, baselines)
                row.update({
                    "model": "meta-llama/Llama-3.3-70B-Instruct",
                    "prompt_id": int(q_idx),
                    "layer": layer,
                    "k": k,
                    "prompt_length": int(payload.get("seq_len") or len(hidden)),
                    "max_triplets_per_selector": int(max_triplets_per_selector),
                })
                selector_rows.append(row)
                if k == 3 and tuples:
                    for trow in triplet_feature_rows(
                        selector,
                        tuples,
                        a,
                        basis,
                        x_proj,
                        meta,
                        neighbors,
                        weights,
                        neighbor_sets,
                        communities,
                        max_rows=max_triplets_per_selector,
                    ):
                        trow.update({"model": row["model"], "prompt_id": int(q_idx), "layer": layer, "k": k})
                        triplet_rows.append(trow)
    return int(q_idx), selector_rows, triplet_rows


def plot_pairwise_vs_kwise(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = df[df["available"] == True].groupby(["k", "selector"], as_index=False)["sign_entropy"].mean()  # noqa: E712
    selectors = [
        "content_only_individual_degree_salience",
        "individual_degree_salience",
        "diffusion_profile_similarity",
        "common_neighbor_two_hop",
        "hidden_state_cosine_cohesion",
        "mutual_attention_avg_pair",
        "mutual_attention_min_pair",
        "triangle_closure_k3",
        "random_control",
        "position_matched_control",
        "norm_matched_control",
    ]
    selectors = [s for s in selectors if s in set(summary["selector"])]
    ranks = sorted(summary["k"].unique())
    fig, axes = plt.subplots(1, len(ranks), figsize=(5.2 * len(ranks), 5.2), constrained_layout=True)
    if len(ranks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ranks):
        g = summary[summary["k"] == k].set_index("selector").reindex(selectors)
        vals = g["sign_entropy"].to_numpy(dtype=float)
        colors = ["#53A6E8" if "control" not in s else "#9AA1AA" for s in selectors]
        ax.barh(range(len(selectors)), vals, color=colors)
        ax.set_yticks(range(len(selectors)))
        ax.set_yticklabels([SELECTOR_LABELS.get(s, s).replace(" / ", "\n") for s in selectors], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("mean sign entropy; lower = cleaner sign ordering")
        ax.set_title(f"k={int(k)}")
        ax.set_xlim(max(0.0, np.nanmin(vals) - 0.02), min(1.0, np.nanmax(vals) + 0.01))
    fig.suptitle("70B GPQA pairwise vs k-wise selector audit")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_triplet_features(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "triangle_closure",
        "common_neighbor_overlap_pair_mean",
        "diffusion_row_similarity",
        "hidden_state_cosine_cohesion",
        "abs_minor",
    ]
    summary = df.groupby("selector", as_index=False)[metrics].mean(numeric_only=True)
    selectors = list(summary.sort_values("abs_minor", ascending=False)["selector"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 5.0), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        vals = summary.set_index("selector").reindex(selectors)[metric].to_numpy(dtype=float)
        ax.barh(range(len(selectors)), vals, color="#53A6E8")
        ax.set_yticks(range(len(selectors)))
        ax.set_yticklabels([s.replace("_", "\n") for s in selectors], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(metric.replace("_", " "), fontsize=9)
    fig.suptitle("k=3 triplet relation features by selector")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_task_audit(df: pd.DataFrame, path: Path) -> None:
    available = df[df["available"] == True].copy()  # noqa: E712
    if available.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    available["task_group"] = available["family"].map(task_group_name)
    available = available[available["task_group"].isin(["analogy", "syllogism"])]
    if available.empty:
        return
    summary = available.groupby(["task_group", "readout_k"], as_index=False)["entropy_gap"].mean()
    ranks = [2, 3, 4]
    groups = ["syllogism", "analogy"]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = np.arange(len(ranks))
    for offset, group in [(-width / 2, "syllogism"), (width / 2, "analogy")]:
        vals = []
        for k in ranks:
            m = summary[(summary["task_group"] == group) & (summary["readout_k"] == k)]["entropy_gap"]
            vals.append(float(m.iloc[0]) if len(m) else np.nan)
        ax.bar(x + offset, vals, width=width, label=group)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ranks])
    ax.set_ylabel("mean random entropy - selected entropy")
    ax.set_title("Analogy vs syllogism, existing task-bank selector")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def task_group_name(family: str) -> str:
    text = str(family)
    if "analogy" in text:
        return "analogy"
    if "syllogism" in text or "binary_relation" in text:
        return "syllogism"
    if "factual" in text:
        return "factual"
    if "theoretical" in text or "integration" in text:
        return "theoretical_integration"
    if "gpqa" in text:
        return "gpqa_multifactor"
    return text


def flatten_task_bank(task_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not task_dir.exists():
        return pd.DataFrame()
    for path in sorted(task_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = d.get("steps") or []
        if not steps:
            continue
        final = steps[-1]
        step_count = len(steps)
        for readout_k, vals in (final.get("k_values") or {}).items():
            if int(readout_k) not in {2, 3, 4, 5}:
                continue
            rows.append({
                "source_file": str(path),
                "source_type": "metrics_only_task_bank",
                "available": True,
                "selector": "original_task_bank_mutual_attention",
                "selector_label": SELECTOR_LABELS["original_task_bank_mutual_attention"],
                "selector_status": "available only for original task-bank run; raw hidden/attention cache not present",
                "family": d.get("family"),
                "task_group": task_group_name(str(d.get("family"))),
                "prompt_id": d.get("prompt_id"),
                "level_k": d.get("level_k"),
                "level_name": d.get("level_name"),
                "schedule_steps": int(step_count),
                "readout_k": int(readout_k),
                "entropy_gap": float(vals.get("entropy_gap", np.nan)),
                "H_selected": float(vals.get("high_attn_entropy", np.nan)),
                "H_control": float(vals.get("random_entropy", np.nan)),
                "n_selected": int(vals.get("n_high_tuples", 0)),
                "n_control": int(vals.get("n_random_tuples", 0)),
                "prompt_text": d.get("prompt_text") or d.get("display_text"),
            })
    task_df = pd.DataFrame(rows)
    if task_df.empty:
        return task_df

    # Make the requested selector audit explicit: only the original selector is
    # available from these summary JSONs. The rest need a raw task cache.
    audit_rows = []
    key_cols = ["family", "task_group", "prompt_id", "level_k", "level_name", "schedule_steps", "readout_k", "prompt_text"]
    base = task_df[task_df["task_group"].isin(["analogy", "syllogism"]) & task_df["readout_k"].isin([2, 3, 4])]
    for _, r in base.iterrows():
        audit_rows.append(r.to_dict())
        for selector in REQUESTED_TASK_SELECTORS:
            if selector == "mutual_attention_min_pair":
                continue
            miss = {col: r[col] for col in key_cols}
            miss.update({
                "source_file": r["source_file"],
                "source_type": "metrics_only_task_bank",
                "available": False,
                "selector": selector,
                "selector_label": SELECTOR_LABELS.get(selector, selector),
                "selector_status": "not rerunnable from summary-only task-bank JSON; requires cached task hidden states and attention",
                "entropy_gap": np.nan,
                "H_selected": np.nan,
                "H_control": np.nan,
                "n_selected": 0,
                "n_control": 0,
            })
            audit_rows.append(miss)
    return pd.DataFrame(audit_rows), task_df


def plot_task_rank_heatmap(profile: pd.DataFrame, path: Path) -> None:
    if profile.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fams = ["factual", "syllogism", "analogy", "theoretical_integration", "gpqa_multifactor"]
    fams = [f for f in fams if f in set(profile["task_group"])]
    ranks = sorted(profile["readout_k"].unique())
    mat = profile.pivot_table(index="task_group", columns="readout_k", values="mean_entropy_gap", aggfunc="mean").reindex(fams).reindex(columns=ranks)
    fig, ax = plt.subplots(figsize=(1.1 * len(ranks) + 4, 0.6 * len(fams) + 2.0), constrained_layout=True)
    vals = mat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"k={k}" for k in ranks])
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f.replace("_", " ") for f in fams])
    ax.set_title("Task family x semantic rank, existing task-bank selector")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean entropy gap")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_task_profile(task_full: pd.DataFrame) -> pd.DataFrame:
    if task_full.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in task_full.groupby(["task_group", "readout_k", "selector"], dropna=False):
        vals = pd.to_numeric(g["entropy_gap"], errors="coerce").dropna().to_numpy(dtype=float)
        t_stat, p_value = (np.nan, 1.0)
        if len(vals) > 1 and np.std(vals, ddof=1) > 0:
            t_stat, p_value = stats.ttest_1samp(vals, popmean=0.0, alternative="greater")
        rows.append({
            "task_group": keys[0],
            "readout_k": int(keys[1]),
            "selector": keys[2],
            "selector_label": SELECTOR_LABELS.get(keys[2], keys[2]),
            "n": int(len(vals)),
            "mean_entropy_gap": float(np.mean(vals)) if len(vals) else np.nan,
            "std_entropy_gap": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_value),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = bh_qvalues(out["p_value"].to_numpy(dtype=float))
    return out


def write_summary(
    path: Path,
    selector_df: pd.DataFrame,
    triplet_df: pd.DataFrame,
    task_audit: pd.DataFrame,
    task_profile: pd.DataFrame,
    layers: Sequence[int],
    ranks: Sequence[int],
) -> None:
    lines: List[str] = []
    lines.append("# Semantic-rank Phase 2 diagnostic summary")
    lines.append("")
    lines.append(f"Scope: 70B GPQA cache, layers {list(layers)}, ranks {list(ranks)}. Task-bank analogy/syllogism artifacts were also inspected.")
    lines.append("")
    lines.append("Terminology: high-degree selection is reported as salience / degree / semantic-hub selection, not relation-free.")
    lines.append("")
    if not selector_df.empty:
        avail = selector_df[selector_df["available"] == True].copy()  # noqa: E712
        mean_table = avail.groupby(["k", "selector"], as_index=False)["sign_entropy"].mean().sort_values(["k", "sign_entropy"])
        lines.append("## 1. Selector readout")
        lines.append("")
        for k in sorted(mean_table["k"].unique()):
            g = mean_table[mean_table["k"] == k].head(4)
            pretty = "; ".join(f"{row.selector} H={row.sign_entropy:.4f}" for row in g.itertuples())
            lines.append(f"- k={int(k)} lowest-entropy selectors: {pretty}.")
        k3 = mean_table[mean_table["k"] == 3]
        if not k3.empty:
            best = k3.iloc[0]
            lines.append(f"- The cleanest k=3 selector in the GPQA cache was `{best['selector']}` with mean sign entropy {best['sign_entropy']:.4f}.")
        lines.append("")
    if not triplet_df.empty:
        tsummary = triplet_df.groupby("selector", as_index=False)[
            ["triangle_closure", "common_neighbor_overlap_pair_mean", "diffusion_row_similarity", "hidden_state_cosine_cohesion", "abs_minor"]
        ].mean(numeric_only=True).sort_values("abs_minor", ascending=False)
        lines.append("## 2. Triplet features")
        lines.append("")
        for row in tsummary.head(5).itertuples():
            lines.append(
                f"- `{row.selector}`: abs minor {row.abs_minor:.6g}, triangle {row.triangle_closure:.6g}, "
                f"diffusion {row.diffusion_row_similarity:.4f}, hidden cohesion {row.hidden_state_cosine_cohesion:.4f}."
            )
        lines.append("")
    lines.append("## 3. Analogy vs syllogism")
    lines.append("")
    if task_audit.empty:
        lines.append("- No relational task-bank metrics were found.")
    else:
        unavailable = int((task_audit["available"] == False).sum())  # noqa: E712
        available = task_audit[task_audit["available"] == True].copy()  # noqa: E712
        lines.append("- The relational task bank is summary-only: it has entropy gaps from the original task-bank selector, but not raw hidden states or attention. Cross-selector reruns for analogy/syllogism therefore require a new raw cache.")
        lines.append(f"- I wrote unavailable rows for the requested non-original selectors instead of fabricating them ({unavailable} rows marked `available=False`).")
        if not available.empty:
            piv = available.groupby(["task_group", "readout_k"], as_index=False)["entropy_gap"].mean()
            for task_group in ["syllogism", "analogy"]:
                vals = piv[piv["task_group"] == task_group]
                if vals.empty:
                    continue
                pretty = ", ".join(f"k={int(r.readout_k)} {r.entropy_gap:+.4f}" for r in vals.itertuples() if int(r.readout_k) in {2, 3, 4})
                lines.append(f"- Existing original-selector {task_group} entropy gaps: {pretty}.")
    lines.append("")
    lines.append("## 4. Task x rank")
    lines.append("")
    if task_profile.empty:
        lines.append("- Task x rank heatmap could not be built because the task-bank profile was empty.")
    else:
        for row in task_profile.sort_values(["task_group", "readout_k"]).itertuples():
            if int(row.readout_k) in {2, 3, 4, 5}:
                lines.append(f"- {row.task_group} k={int(row.readout_k)}: mean gap {row.mean_entropy_gap:+.4f}, n={int(row.n)}, q={row.q_value:.3g}.")
    lines.append("")
    lines.append("## 5. Safest paper claim")
    lines.append("")
    lines.append(
        "The degree/salience result shows that direct pairwise mutual attention is not the unique selector of the Plucker effect. "
        "The stronger current claim is that semantic-rank geometry is expressed among salience-conditioned semantic hubs and k-wise contextual population structure, "
        "while direct pairwise attention is only one imperfect proxy for relation. The analogy k=3 cross-selector survival question remains open until the analogy/syllogism task bank is recached with raw hidden states and attention."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    layers = parse_int_list(args.layers, [50, 55, 60])
    ranks = parse_int_list(args.ranks, [2, 3, 4])
    cache_dir = Path(args.input_cache_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tuple_cfg = TupleConfig(
        n_hub_tokens=args.n_hub_tokens,
        n_selected_tuples=args.n_tuples,
        n_control_tuples=args.n_tuples,
        n_candidates=args.n_candidates,
        min_gap=args.min_gap,
        position_bins=args.position_bins,
        norm_bins=args.norm_bins,
    )

    prompt_ids = cache_prompt_ids(cache_dir)
    if args.max_prompts is not None:
        prompt_ids = prompt_ids[: int(args.max_prompts)]
    selector_rows: List[Dict[str, object]] = []
    triplet_rows: List[Dict[str, object]] = []
    jobs = [
        (
            str(cache_dir),
            int(q_idx),
            list(layers),
            list(ranks),
            tuple_cfg,
            int(args.projection_seed),
            int(args.proj_dim),
            int(args.graph_top_neighbors),
            int(args.max_triplets_per_selector),
        )
        for q_idx in prompt_ids
    ]
    if int(args.workers) <= 1:
        for pos, job in enumerate(jobs, start=1):
            q_idx, srows, trows = process_prompt_job(job)
            selector_rows.extend(srows)
            triplet_rows.extend(trows)
            if pos % max(1, int(args.progress_every)) == 0:
                print(f"[live] GPQA selector audit processed {pos}/{len(jobs)} prompts; latest q_idx={q_idx}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(process_prompt_job, job) for job in jobs]
            for pos, fut in enumerate(as_completed(futures), start=1):
                q_idx, srows, trows = fut.result()
                selector_rows.extend(srows)
                triplet_rows.extend(trows)
                if pos % max(1, int(args.progress_every)) == 0:
                    print(
                        f"[live] GPQA selector audit processed {pos}/{len(jobs)} prompts; latest q_idx={q_idx}; "
                        f"rows={len(selector_rows)} triplets={len(triplet_rows)}",
                        flush=True,
                    )

    selector_df = pd.DataFrame(selector_rows)
    selector_path = results_dir / "pairwise_vs_kwise_selectors_70b.csv"
    selector_df.to_csv(selector_path, index=False)
    plot_pairwise_vs_kwise(selector_df, figures_dir / "fig_pairwise_vs_kwise_selectors_70b.png")

    triplet_df = pd.DataFrame(triplet_rows)
    triplet_path = results_dir / "triplet_relation_features_k3_70b.csv"
    triplet_df.to_csv(triplet_path, index=False)
    plot_triplet_features(triplet_df, figures_dir / "fig_triplet_relation_features_k3_70b.png")

    task_result = flatten_task_bank(Path(args.task_bank_dir).expanduser().resolve())
    if isinstance(task_result, tuple):
        task_audit, task_full = task_result
    else:
        task_audit, task_full = pd.DataFrame(), task_result
    task_audit_path = results_dir / "analogy_syllogism_selector_audit.csv"
    task_audit.to_csv(task_audit_path, index=False)
    plot_task_audit(task_audit, figures_dir / "fig_analogy_syllogism_k3_by_selector.png")

    task_profile = summarize_task_profile(task_full)
    task_profile_path = results_dir / "task_rank_semantic_profile.csv"
    task_profile.to_csv(task_profile_path, index=False)
    plot_task_rank_heatmap(task_profile, figures_dir / "fig_task_rank_semantic_profile.png")

    write_summary(
        results_dir / "semantic_rank_diagnostic_summary.md",
        selector_df,
        triplet_df,
        task_audit,
        task_profile,
        layers,
        ranks,
    )
    print(f"Wrote {selector_path}")
    print(f"Wrote {triplet_path}")
    print(f"Wrote {task_audit_path}")
    print(f"Wrote {task_profile_path}")
    print(f"Wrote {results_dir / 'semantic_rank_diagnostic_summary.md'}")


if __name__ == "__main__":
    main()
