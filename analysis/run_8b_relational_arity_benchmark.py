#!/usr/bin/env python3
"""8B controlled relational-arity benchmark.

This intentionally avoids broad semantic prompt families.  Each prompt is a
compact relational-complexity puzzle with repeated facts of one arity r, plus a
single target fact.  The geometry analysis can therefore use many true r-way
argument tuples per prompt instead of a single noisy tuple.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    answer_margin,
    answer_scores_from_logits,
    answer_token_candidates,
    bh_qvalues,
    bidirectional_attention,
    compute_basis,
    compute_minors,
    deterministic_projection,
    iter_cached_payloads,
    load_hf_model_and_tokenizer,
    parse_int_list,
    token_classes_from_ids,
    write_json,
)
from analysis.run_8b_gpqa_highk_depth_audit import (  # noqa: E402
    SELECTOR_LABELS as GPQA_SELECTOR_LABELS,
    build_selector_prefixes,
)
from analysis.run_semantic_rank_phase2 import (  # noqa: E402
    bidir_no_self,
    norm_matched_from_pool,
    position_matched_from_pool,
    random_tuples_from_pool,
)


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LAYERS = [0, 5, 10, 15, 20, 25, 30, 31]
DEFAULT_RANKS = [1, 2, 3, 4, 5, 6, 7]
SELECTOR_LABELS = {
    **GPQA_SELECTOR_LABELS,
    "ground_truth_arguments": "ground-truth argument tuples",
    "argument_plus_predicate": "argument + predicate tuples",
    "scrambled_argument_control": "scrambled-argument control",
}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def npz_is_readable(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            return "input_ids" in z.files and len(z.files) > 1
    except Exception:
        return False


def safe_save_npz(path: Path, payload: Mapping[str, np.ndarray], attempts: int = 5) -> None:
    """Save through a temp file so network-volume write hiccups do not poison cache rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    last_exc: Optional[BaseException] = None
    for attempt in range(int(attempts)):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            with tmp_path.open("wb") as f:
                np.savez(f, **payload)
                f.flush()
                os.fsync(f.fileno())
            if not npz_is_readable(tmp_path):
                raise OSError(f"temporary npz failed validation: {tmp_path}")
            os.replace(tmp_path, path)
            return
        except (OSError, ValueError) as exc:
            last_exc = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
    raise OSError(f"failed to save valid npz after {attempts} attempts: {path}") from last_exc


def tuple_choice(args: Sequence[str]) -> str:
    return "(" + ", ".join(args) + ")"


def predicate_for_arity(r: int) -> str:
    return {
        1: "UNARYTAG",
        2: "PAIRLINK",
        3: "TRIADMAP",
        4: "QUADBIND",
        5: "PENTASYS",
        6: "HEXANET",
    }[int(r)]


def relation_expr(predicate: str, args: Sequence[str]) -> str:
    return f"{predicate}({', '.join(args)})"


def make_args(r: int, prompt_i: int, fact_i: int) -> List[str]:
    # Unique only within a prompt is enough. Keep labels tiny so r=6 prompts do
    # not get truncated before the answer options.
    role_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{role_letters[j]}{fact_i}" for j in range(int(r))]


def make_distractors(r: int, prompt_i: int, n: int = 8) -> List[str]:
    return [f"Z{i}" for i in range(int(n))]


def rotated_distractor_choices(correct_args: Sequence[str], distractors: Sequence[str]) -> List[str]:
    r = len(correct_args)
    if r == 1:
        return [tuple_choice([correct_args[0]])] + [tuple_choice([d]) for d in distractors[:3]]
    variants = [list(correct_args)]
    # Candidate distractors. Some operations collapse for r=2, so de-duplicate
    # before assigning A/B/C/D.
    variants.append(list(reversed(correct_args)))
    swap = list(correct_args)
    swap[0], swap[1] = swap[1], swap[0]
    variants.append(swap)
    repl_last = list(correct_args)
    repl_last[-1] = distractors[0]
    variants.append(repl_last)
    repl_first = list(correct_args)
    repl_first[0] = distractors[1]
    variants.append(repl_first)
    rotate = list(correct_args[1:]) + [correct_args[0]]
    variants.append(rotate)
    seen = set()
    out = []
    for var in variants:
        text = tuple_choice(var)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) == 4:
            break
    if len(out) < 4:
        for d in distractors:
            filler = list(correct_args)
            filler[len(out) % r] = d
            text = tuple_choice(filler)
            if text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) == 4:
                break
    return out[:4]


def build_task_bank(n_per_arity: int, n_relation_instances: int, seed: int = 40577) -> List[dict]:
    rng = np.random.default_rng(int(seed))
    rows: List[dict] = []
    for r in range(1, 7):
        predicate = predicate_for_arity(r)
        for i in range(int(n_per_arity)):
            answer_letter = ["A", "B", "C", "D"][i % 4]
            relations = []
            for fact_i in range(int(n_relation_instances)):
                args = make_args(r, i, fact_i)
                relations.append(
                    {
                        "fact_label": f"F{fact_i + 1:02d}",
                        "predicate": predicate,
                        "arguments": args,
                        "relation_expr": relation_expr(predicate, args),
                    }
                )
            target_idx = int(rng.integers(0, len(relations)))
            target = relations[target_idx]
            distractors = make_distractors(r, i)
            choices_raw = rotated_distractor_choices(target["arguments"], distractors)
            # Put the correct option in the balanced slot.
            ordered_letters = ["A", "B", "C", "D"]
            choices = {}
            distractor_iter = iter(choices_raw[1:])
            for letter in ordered_letters:
                choices[letter] = choices_raw[0] if letter == answer_letter else next(distractor_iter)
            facts = "\n".join(f"  {rel['fact_label']}: {rel['relation_expr']}" for rel in relations)
            prompt = (
                "This is a controlled relational-complexity puzzle. "
                "Every fact below has one predicate and an ordered argument tuple.\n\n"
                f"Relation arity: {r}\n"
                f"Facts:\n{facts}\n\n"
                f"Target fact: {target['fact_label']} = {target['relation_expr']}\n"
                "Which option lists the target fact's arguments in the exact same order?"
            )
            rows.append(
                {
                    "prompt_id": f"rel_arity_r{r}_{i:03d}",
                    "task_family": "standardized_relational_arity",
                    "style_condition": "symbolic_controlled",
                    "relation_arity": int(r),
                    "relational_order_target": int(r),
                    "prompt": prompt,
                    "choices": choices,
                    "answer": answer_letter,
                    "entities": [arg for rel in relations for arg in rel["arguments"]],
                    "distractor_entities": distractors,
                    "relations": relations,
                    "target_relation_index": target_idx,
                    "target_relation": target,
                    "metadata_schema": "argument/predicate spans are added after tokenization in the cache metadata",
                }
            )
    return rows


def format_relational_prompt(row: Mapping, tokenizer) -> str:
    choices = row.get("choices") or {}
    prompt = f"Question: {str(row.get('prompt', '')).strip()}\n\nChoices:\n"
    for letter in ["A", "B", "C", "D"]:
        prompt += f"  ({letter}) {choices.get(letter, '')}\n"
    prompt += "\nAnswer: The correct answer is ("
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def char_span_to_token_span(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    hits = []
    for idx, (a, b) in enumerate(offsets):
        if int(a) == int(b) == 0:
            continue
        if int(b) <= int(start) or int(a) >= int(end):
            continue
        hits.append(int(idx))
    return hits


def first_token(tokens: Sequence[int]) -> Optional[int]:
    return int(tokens[0]) if tokens else None


def locate_relation_spans(full_prompt: str, row: Mapping, offsets: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    relations_out = []
    cursor = full_prompt.find("Facts:")
    if cursor < 0:
        cursor = 0
    for rel in row.get("relations", []):
        expr = str(rel.get("relation_expr"))
        pos = full_prompt.find(expr, cursor)
        if pos < 0:
            pos = full_prompt.find(expr)
        expr_end = pos + len(expr) if pos >= 0 else -1
        pred = str(rel.get("predicate"))
        pred_pos = full_prompt.find(pred, pos, expr_end) if pos >= 0 else -1
        pred_tokens = char_span_to_token_span(offsets, pred_pos, pred_pos + len(pred)) if pred_pos >= 0 else []
        arg_spans = []
        arg_reps = []
        local_cursor = pred_pos + len(pred) if pred_pos >= 0 else pos
        for arg in rel.get("arguments", []):
            arg = str(arg)
            arg_pos = full_prompt.find(arg, local_cursor, expr_end) if expr_end >= 0 else -1
            toks = char_span_to_token_span(offsets, arg_pos, arg_pos + len(arg)) if arg_pos >= 0 else []
            arg_spans.append({"text": arg, "char_span": [arg_pos, arg_pos + len(arg)] if arg_pos >= 0 else None, "token_span": toks})
            arg_reps.append(first_token(toks))
            if arg_pos >= 0:
                local_cursor = arg_pos + len(arg)
        relations_out.append(
            {
                "fact_label": rel.get("fact_label"),
                "predicate": pred,
                "relation_expr": expr,
                "relation_char_span": [pos, expr_end] if pos >= 0 else None,
                "predicate_token_span": pred_tokens,
                "predicate_rep_token": first_token(pred_tokens),
                "argument_token_spans": arg_spans,
                "argument_rep_tokens": arg_reps,
            }
        )
        if expr_end > cursor:
            cursor = expr_end
    distractor_spans = []
    for ent in row.get("distractor_entities", []):
        ent = str(ent)
        pos = full_prompt.find(ent)
        toks = char_span_to_token_span(offsets, pos, pos + len(ent)) if pos >= 0 else []
        distractor_spans.append({"text": ent, "char_span": [pos, pos + len(ent)] if pos >= 0 else None, "token_span": toks, "rep_token": first_token(toks)})
    target_idx = int(row.get("target_relation_index", 0))
    return {
        "relation_instances": relations_out,
        "target_relation_index": target_idx,
        "target_relation_instance": relations_out[target_idx] if 0 <= target_idx < len(relations_out) else None,
        "distractor_token_spans": distractor_spans,
    }


def capture_cache(args: argparse.Namespace, task_rows: Sequence[Mapping]) -> None:
    import torch

    layers = parse_int_list(args.layers, DEFAULT_LAYERS)
    out_dir = Path(args.output_cache_dir).expanduser().resolve()
    arrays_dir = out_dir / "arrays"
    prompts_dir = out_dir / "prompts"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    hidden_dtype = np.float16 if args.store_hidden_dtype == "float16" else np.float32
    attn_dtype = np.float16 if args.store_attn_dtype == "float16" else np.float32
    tokenizer, model = load_hf_model_and_tokenizer(
        model_id=args.model_load_path or args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only),
        device_map=args.device_map,
        load_in_4bit=not bool(args.no_4bit),
    )
    answer_ids = {letter: answer_token_candidates(tokenizer, letter) for letter in ["A", "B", "C", "D"]}
    captured = []
    skipped = []
    t0 = time.time()
    for pos, row in enumerate(task_rows, start=1):
        idx = pos - 1
        arr_path = arrays_dir / f"prompt_{idx:04d}.npz"
        meta_path = prompts_dir / f"prompt_{idx:04d}.json"
        if args.skip_existing and arr_path.exists() and meta_path.exists() and arr_path.stat().st_size > 1024 and npz_is_readable(arr_path):
            captured.append(idx)
            skipped.append(idx)
            continue
        prompt_text = format_relational_prompt(row, tokenizer)
        enc_kwargs = dict(return_tensors="pt", truncation=True, max_length=int(args.max_length))
        try:
            enc = tokenizer(prompt_text, return_offsets_mapping=True, **enc_kwargs)
            offsets = [(int(a), int(b)) for a, b in enc.pop("offset_mapping")[0].detach().cpu().tolist()]
        except Exception:
            enc = tokenizer(prompt_text, **enc_kwargs)
            offsets = []
        input_ids = enc["input_ids"][0].detach().cpu().numpy().astype(int)
        model_inputs = {name: tensor.to(model.device) for name, tensor in enc.items()}
        with torch.no_grad():
            out = model(**model_inputs, output_hidden_states=True, output_attentions=True)
        correct_idx = ["A", "B", "C", "D"].index(str(row.get("answer")))
        scores = answer_scores_from_logits(out.logits, answer_ids)
        behavior = answer_margin(scores, correct_idx)
        arr_payload = {"input_ids": input_ids.astype(np.int32)}
        for layer in layers:
            if int(layer) + 1 < len(out.hidden_states):
                arr_payload[f"hidden_L{int(layer)}"] = out.hidden_states[int(layer) + 1][0].detach().float().cpu().numpy().astype(hidden_dtype)
            if int(layer) < len(out.attentions):
                arr_payload[f"bidir_L{int(layer)}"] = bidirectional_attention(out.attentions[int(layer)][0].detach().float().cpu().numpy()).astype(attn_dtype)
        safe_save_npz(arr_path, arr_payload)
        try:
            token_strings = tokenizer.convert_ids_to_tokens([int(x) for x in input_ids])
        except Exception:
            token_strings = [str(int(x)) for x in input_ids]
        span_metadata = locate_relation_spans(prompt_text, row, offsets)
        meta = {
            "q_idx": int(idx),
            "prompt_id": row.get("prompt_id"),
            "item": row,
            "seq_len": int(enc["input_ids"].shape[1]),
            "input_ids": [int(x) for x in input_ids],
            "token_strings": token_strings,
            "token_classes": token_classes_from_ids(tokenizer, input_ids),
            "span_metadata": span_metadata,
            "answer_scores": scores,
            **behavior,
        }
        write_json(meta_path, meta)
        captured.append(idx)
        del out, model_inputs, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.live_every and (pos % max(1, int(args.live_every)) == 0 or pos == len(task_rows)):
            print(f"[arity-capture] {pos}/{len(task_rows)} prompts in {time.time() - t0:.1f}s latest={row.get('prompt_id')}", flush=True)
    manifest = {
        "cache_format": "relational_arity_hidden_bidir_attention_npz_v1",
        "model_id": args.model_id,
        "model_load_path": args.model_load_path,
        "layers": layers,
        "prompt_ids": sorted(captured),
        "max_length": int(args.max_length),
        "hidden_dtype": args.store_hidden_dtype,
        "attention_dtype": args.store_attn_dtype,
        "projection_dim_for_downstream": DEFAULT_PROJ_DIM,
        "task_bank": str(Path(args.task_bank).expanduser().resolve()),
        "arrays_dir": str(arrays_dir),
        "prompts_dir": str(prompts_dir),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[arity-capture] wrote {out_dir / 'manifest.json'} captured={len(captured)} skipped={len(skipped)}", flush=True)


def entropy_from_minors(minors: Sequence[float]) -> Tuple[float, int]:
    vals = np.asarray([float(m) for m in minors if np.isfinite(m)], dtype=float)
    vals = vals[np.abs(vals) > 0.0]
    if vals.size < 2:
        return 0.0, int(vals.size)
    p = float(np.mean(vals > 0))
    if p <= 0.0 or p >= 1.0:
        return 0.0, int(vals.size)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))), int(vals.size)


def valid_tuple(tup: Sequence[Optional[int]], k: int, seq_len: int) -> Optional[Tuple[int, ...]]:
    vals = [int(x) for x in tup if x is not None and 0 <= int(x) < int(seq_len)]
    if len(vals) != int(k) or len(set(vals)) != len(vals):
        return None
    return tuple(vals)


def arity_true_tuples(payload: Mapping, k: int, selector: str) -> List[Tuple[int, ...]]:
    span_meta = payload.get("span_metadata") or {}
    relations = span_meta.get("relation_instances") or []
    item = payload.get("item") or {}
    arity = int(item.get("relation_arity") or 0)
    seq_len = int(payload.get("seq_len") or 0)
    out: List[Tuple[int, ...]] = []
    if selector == "ground_truth_arguments" and int(k) == arity:
        for rel in relations:
            tup = valid_tuple(rel.get("argument_rep_tokens") or [], k, seq_len)
            if tup is not None:
                out.append(tup)
    elif selector == "argument_plus_predicate" and int(k) == arity + 1:
        for rel in relations:
            tup = valid_tuple([rel.get("predicate_rep_token")] + list(rel.get("argument_rep_tokens") or []), k, seq_len)
            if tup is not None:
                out.append(tup)
    elif selector == "scrambled_argument_control" and int(k) == arity:
        role_matrix = [list(rel.get("argument_rep_tokens") or []) for rel in relations]
        if role_matrix and all(len(row) == arity for row in role_matrix):
            n = len(role_matrix)
            for i in range(n):
                scrambled = [role_matrix[(i + role + 1) % n][role] for role in range(arity)]
                tup = valid_tuple(scrambled, k, seq_len)
                if tup is not None:
                    out.append(tup)
    return out


def matched_controls(
    source_tuples: Sequence[Tuple[int, ...]],
    seq_len: int,
    norms: np.ndarray,
    seed_base: int,
    min_gap: int,
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    pool = list(range(int(seq_len)))
    n = len(source_tuples)
    k = len(source_tuples[0]) if source_tuples else 0
    random = random_tuples_from_pool(pool, k, n, seed_base + 11, min_gap, exclude=source_tuples)
    pos = position_matched_from_pool(source_tuples, pool, int(seq_len), seed_base + 12, 10, min_gap)[:n]
    norm = norm_matched_from_pool(source_tuples, pool, norms, seed_base + 13, 10, min_gap)[:n]
    return random, pos, norm


def analyze_prompt_rows(worker_args: Tuple[str, List[int], List[int], int, int, int, int, int, int, int]) -> List[Dict[str, object]]:
    (
        cache_dir_s,
        layers,
        ranks,
        prompt_id,
        proj_dim,
        projection_seed,
        control_tuple_budget,
        n_candidates,
        n_hub_tokens,
        min_gap,
    ) = worker_args
    cache_dir = Path(cache_dir_s).expanduser().resolve()
    rows: List[Dict[str, object]] = []
    for payload in iter_cached_payloads(cache_dir, layers=layers, prompt_ids=[prompt_id]):
        item = payload.get("item") or {}
        arity = int(item.get("relation_arity") or 0)
        q_idx = int(payload["q_idx"])
        for layer in layers:
            if layer not in payload["hidden_by_layer"] or layer not in payload["attn_by_layer"]:
                continue
            hidden = payload["hidden_by_layer"][layer].astype(np.float32, copy=False)
            attn = bidir_no_self(payload["attn_by_layer"][layer])
            proj = deterministic_projection(hidden.shape[1], int(proj_dim), int(projection_seed))
            x_proj = hidden @ proj
            norms = np.sqrt(np.sum(x_proj * x_proj, axis=1))
            for k in ranks:
                basis = compute_basis(x_proj, int(k), center=False)
                seed_base = q_idx * 100000 + int(layer) * 100 + int(k)
                selector_prefixes = build_selector_prefixes(
                    attn=attn,
                    norms=norms,
                    k=int(k),
                    max_budget=int(control_tuple_budget),
                    n_candidates=int(n_candidates),
                    n_hub_tokens=int(n_hub_tokens),
                    min_gap=int(min_gap),
                    position_bins=10,
                    norm_bins=10,
                    seed_base=seed_base,
                )
                selector_map = {
                    "ground_truth_arguments": arity_true_tuples(payload, int(k), "ground_truth_arguments"),
                    "argument_plus_predicate": arity_true_tuples(payload, int(k), "argument_plus_predicate"),
                    "scrambled_argument_control": arity_true_tuples(payload, int(k), "scrambled_argument_control"),
                    "degree_salience": selector_prefixes.get("degree_salience", [])[: int(control_tuple_budget)],
                    "avg_pair_attention": selector_prefixes.get("avg_pair_attention", [])[: int(control_tuple_budget)],
                    "k_clique_all_pairs_attention": selector_prefixes.get("k_clique_all_pairs_attention", [])[: int(control_tuple_budget)] if int(k) >= 4 else [],
                }
                for selector, tuples in selector_map.items():
                    if not tuples:
                        continue
                    selector_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(selector))
                    random_tuples, pos_tuples, norm_tuples = matched_controls(
                        tuples,
                        int(payload.get("seq_len") or hidden.shape[0]),
                        norms,
                        seed_base + selector_seed % 10000,
                        int(min_gap),
                    )
                    selected_minors = compute_minors(basis, tuples)
                    random_minors = compute_minors(basis, random_tuples)
                    pos_minors = compute_minors(basis, pos_tuples)
                    norm_minors = compute_minors(basis, norm_tuples)
                    h_sel, n_sel = entropy_from_minors(selected_minors)
                    h_rand, n_rand = entropy_from_minors(random_minors)
                    h_pos, n_pos = entropy_from_minors(pos_minors)
                    h_norm, n_norm = entropy_from_minors(norm_minors)
                    rows.append(
                        {
                            "model": MODEL_ID,
                            "benchmark": "controlled_relational_arity",
                            "prompt_id": q_idx,
                            "task_prompt_id": item.get("prompt_id"),
                            "relation_arity": arity,
                            "layer": int(layer),
                            "k": int(k),
                            "selector": selector,
                            "selector_label": SELECTOR_LABELS.get(selector, selector),
                            "n_selector_tuples": int(n_sel),
                            "n_random_tuples": int(n_rand),
                            "n_position_tuples": int(n_pos),
                            "n_norm_tuples": int(n_norm),
                            "H_selector": h_sel,
                            "H_random": h_rand,
                            "H_position": h_pos,
                            "H_norm": h_norm,
                            "random_minus_selector_entropy_gap": float(h_rand - h_sel),
                            "position_minus_selector_entropy_gap": float(h_pos - h_sel),
                            "norm_minus_selector_entropy_gap": float(h_norm - h_sel),
                            "selector_minus_scrambled_relevant": selector in {"ground_truth_arguments", "argument_plus_predicate"},
                            "pred_letter": payload.get("pred_letter"),
                            "correct_letter": payload.get("correct_letter"),
                            "is_correct": payload.get("is_correct"),
                        }
                    )
    return rows


def analyze_cache(args: argparse.Namespace) -> None:
    cache_dir = Path(args.output_cache_dir).expanduser().resolve()
    layers = parse_int_list(args.layers, DEFAULT_LAYERS)
    ranks = parse_int_list(args.ranks, DEFAULT_RANKS)
    prompt_ids = sorted(int(p.stem.split("_")[-1]) for p in (cache_dir / "arrays").glob("prompt_*.npz"))
    worker_args = [
        (
            str(cache_dir),
            list(map(int, layers)),
            list(map(int, ranks)),
            int(prompt_id),
            int(args.proj_dim),
            int(args.projection_seed),
            int(args.control_tuple_budget),
            int(args.n_candidates),
            int(args.n_hub_tokens),
            int(args.min_gap),
        )
        for prompt_id in prompt_ids
    ]
    all_rows: List[Dict[str, object]] = []
    workers = max(1, int(getattr(args, "analysis_workers", 1)))
    if workers > 1:
        print(f"[arity-analysis] parallel workers={workers} prompts={len(worker_args)}", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(analyze_prompt_rows, wa) for wa in worker_args]
            for pos, fut in enumerate(as_completed(futures), start=1):
                all_rows.extend(fut.result())
                if args.live_every and (pos % max(1, int(args.live_every)) == 0 or pos == len(prompt_ids)):
                    print(f"[arity-analysis] analyzed {pos}/{len(prompt_ids)} cached prompts rows={len(all_rows)}", flush=True)
    else:
        for pos, wa in enumerate(worker_args, start=1):
            all_rows.extend(analyze_prompt_rows(wa))
            if args.live_every and (pos % max(1, int(args.live_every)) == 0 or pos == len(prompt_ids)):
                print(f"[arity-analysis] analyzed {pos}/{len(prompt_ids)} cached prompts rows={len(all_rows)}", flush=True)
    rows_df = pd.DataFrame(all_rows)
    if not rows_df.empty:
        rows_df = rows_df.sort_values(["prompt_id", "layer", "k", "selector"]).reset_index(drop=True)
    results_dir = Path(args.results_dir).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows_path = results_dir / "8b_relational_arity_rows.csv"
    stats_path = results_dir / "8b_relational_arity_stats.csv"
    rows_df.to_csv(rows_path, index=False)
    stats_df = summarize_arity(rows_df)
    stats_df.to_csv(stats_path, index=False)
    plot_arity_heatmap(stats_df, figures_dir / "fig_8b_relation_arity_by_k_heatmap.png")
    plot_ground_truth_vs_controls(stats_df, figures_dir / "fig_8b_ground_truth_arguments_vs_controls.png")
    write_arity_summary(stats_df, results_dir / "8b_relational_arity_summary.md")
    print(f"[arity-analysis] wrote {rows_path}", flush=True)
    print(f"[arity-analysis] wrote {stats_path}", flush=True)


def bootstrap_ci(vals: Sequence[float], seed: int = 405078, n_boot: int = 1000) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    boot = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(int(n_boot))]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize_arity(rows_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["relation_arity", "layer", "k", "selector", "selector_label"]
    metrics = ["random_minus_selector_entropy_gap", "position_minus_selector_entropy_gap", "norm_minus_selector_entropy_gap"]
    rows = []
    for keys, g in rows_df.groupby(group_cols, dropna=False):
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_prompts"] = int(g["prompt_id"].nunique())
        row["mean_H_selector"] = float(pd.to_numeric(g["H_selector"], errors="coerce").mean())
        row["mean_H_random"] = float(pd.to_numeric(g["H_random"], errors="coerce").mean())
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"mean_{metric}"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_ci(vals)
            row[f"{metric}_positive_fraction"] = float(np.mean(vals > 0)) if vals.size else np.nan
            if vals.size > 1 and float(np.std(vals, ddof=1)) > 0:
                t_stat, p_value = stats.ttest_1samp(vals, popmean=0.0, alternative="greater")
                row[f"{metric}_t_stat"] = float(t_stat)
                row[f"{metric}_p_greater"] = float(p_value)
            else:
                row[f"{metric}_t_stat"] = np.nan
                row[f"{metric}_p_greater"] = 1.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for metric in metrics:
            out[f"{metric}_q_value"] = bh_qvalues(out[f"{metric}_p_greater"].to_numpy(dtype=float))
    return out


def plot_arity_heatmap(stats_df: pd.DataFrame, out_path: Path) -> None:
    sub = stats_df[stats_df["selector"].isin(["ground_truth_arguments", "argument_plus_predicate", "degree_salience", "avg_pair_attention", "k_clique_all_pairs_attention"])].copy()
    if sub.empty:
        return
    best = sub.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).groupby(["relation_arity", "k"], as_index=False).first()
    piv = best.pivot_table(index="relation_arity", columns="k", values="mean_random_minus_selector_entropy_gap", aggfunc="max").sort_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"k={int(x)}" for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"r={int(x)}" for x in piv.index])
    ax.set_xlabel("Pluecker rank k")
    ax.set_ylabel("relation arity r")
    ax.set_title("8B controlled relational-arity: best selector entropy gap")
    for i, r in enumerate(piv.index):
        for j, k in enumerate(piv.columns):
            val = piv.loc[r, k]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > np.nanmax(piv.to_numpy()) * 0.45 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="random - selector entropy gap")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_ground_truth_vs_controls(stats_df: pd.DataFrame, out_path: Path) -> None:
    sub = stats_df[stats_df["selector"].isin(["ground_truth_arguments", "scrambled_argument_control"])].copy()
    if sub.empty:
        return
    # For each arity, show the target k=r, best over layers.
    rows = []
    for arity in sorted(sub["relation_arity"].unique()):
        for selector in ["ground_truth_arguments", "scrambled_argument_control"]:
            g = sub[(sub["relation_arity"] == arity) & (sub["k"] == arity) & (sub["selector"] == selector)]
            if g.empty:
                continue
            rows.append(g.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).iloc[0])
    if not rows:
        return
    best = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    arities = sorted(best["relation_arity"].unique())
    x = np.arange(len(arities), dtype=float)
    width = 0.36
    for off, selector, color in [(-width / 2, "ground_truth_arguments", "#3b82f6"), (width / 2, "scrambled_argument_control", "#f59e0b")]:
        vals = []
        for arity in arities:
            g = best[(best["relation_arity"] == arity) & (best["selector"] == selector)]
            vals.append(float(g["mean_random_minus_selector_entropy_gap"].iloc[0]) if not g.empty else np.nan)
        ax.bar(x + off, vals, width=width, label=SELECTOR_LABELS[selector], color=color, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"r=k={int(r)}" for r in arities])
    ax.set_ylabel("random - selector entropy gap")
    ax.set_title("Ground-truth argument tuples vs scrambled controls at k=r")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_arity_summary(stats_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# 8B Relational-Arity Benchmark Summary",
        "",
        "This run uses controlled Relational Complexity Theory style prompts, not the earlier broad semantic ladder. Each prompt contains repeated facts of one explicit arity r, so ground-truth r-way argument tuples can be compared to random/position/norm and scrambled-argument controls.",
        "",
        "## Best Rows By Arity",
        "",
    ]
    if stats_df.empty:
        lines.append("No stats rows were produced.")
    else:
        sub = stats_df.sort_values("mean_random_minus_selector_entropy_gap", ascending=False)
        for arity, g in sub.groupby("relation_arity"):
            best = g.iloc[0]
            target = g[(g["k"] == arity) & (g["selector"] == "ground_truth_arguments")]
            gt = target.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).iloc[0] if not target.empty else None
            lines.append(
                f"- r={int(arity)} best overall: k={int(best['k'])}, layer={int(best['layer'])}, selector={best['selector_label']}, "
                f"gap={float(best['mean_random_minus_selector_entropy_gap']):+.4f}, q={float(best['random_minus_selector_entropy_gap_q_value']):.3g}."
            )
            if gt is not None:
                lines.append(
                    f"  Ground-truth at k=r: layer={int(gt['layer'])}, gap={float(gt['mean_random_minus_selector_entropy_gap']):+.4f}, "
                    f"q={float(gt['random_minus_selector_entropy_gap_q_value']):.3g}, positive={float(gt['random_minus_selector_entropy_gap_positive_fraction']):.3f}."
                )
    lines.extend(
        [
            "",
            "## Claim Guardrail",
            "",
            "This benchmark is the clean arity test. If high arities only appear under salience selectors but not ground-truth argument tuples, the claim should be phrased as high-k structure among attention-salient populations rather than direct relation-argument geometry.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["generate", "capture", "analyze", "all"], default="all")
    p.add_argument("--task-bank", default=str(ROOT / "data" / "relational_arity_benchmark_prompts.jsonl"))
    p.add_argument("--n-per-arity", type=int, default=100)
    p.add_argument("--n-relation-instances", type=int, default=16)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--model-load-path", default=None)
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    p.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--output-cache-dir", default=str(ROOT / "results" / "cache_8b_relational_arity_benchmark"))
    p.add_argument("--store-hidden-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--store-attn-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--projection-seed", type=int, default=42)
    p.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    p.add_argument("--control-tuple-budget", type=int, default=20)
    p.add_argument("--n-hub-tokens", type=int, default=70)
    p.add_argument("--n-candidates", type=int, default=24000)
    p.add_argument("--min-gap", type=int, default=2)
    p.add_argument("--analysis-workers", type=int, default=1)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    p.add_argument("--live-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task_path = Path(args.task_bank).expanduser().resolve()
    if args.mode in {"generate", "all"} or not task_path.exists():
        rows = build_task_bank(args.n_per_arity, args.n_relation_instances)
        write_jsonl(task_path, rows)
        print(f"[arity] wrote task bank {task_path} rows={len(rows)}", flush=True)
    else:
        rows = read_jsonl(task_path)
    if args.mode in {"capture", "all"}:
        capture_cache(args, rows)
    if args.mode in {"analyze", "all"}:
        analyze_cache(args)


if __name__ == "__main__":
    main()
