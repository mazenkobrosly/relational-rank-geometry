#!/usr/bin/env python3
"""Cache-only projection-dimension/seed audit for headline arity rows.

This is intentionally dependency-light for pod images that only have numpy.
It uses saved hidden-state caches, held-out odd prompt IDs, and dev-selected
layers from the final layer-selection audit.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


MODELS = {
    "8B": {
        "cache": "/workspace/grassmannian_8b_controls/results/cache_8b_relational_arity_benchmark",
        "layers": {3: 25, 4: 20, 5: 20, 6: 20},
    },
    "70B": {
        "cache": "/workspace/grassmannian_70b_controls/results/cache_70b_relational_arity_benchmark",
        "layers": {3: 55, 4: 60, 5: 20, 6: 40},
    },
    "405B": {
        "cache": "/workspace/grassmannian_405b_controls/results/cache_405b_relational_arity_benchmark",
        "layers": {3: 30, 4: 30, 5: 30, 6: 40},
    },
}


def sign_entropy_from_signs(vals: np.ndarray) -> tuple[float, int, int, int]:
    signs = np.asarray(vals, dtype=np.float64)
    signs = signs[np.isfinite(signs)]
    signs = signs[signs != 0]
    n = int(signs.size)
    if n == 0:
        return math.nan, 0, 0, 0
    pos = int(np.sum(signs > 0))
    neg = int(np.sum(signs < 0))
    h = 0.0
    for count in (pos, neg):
        if count:
            p = count / n
            h -= p * math.log2(p)
    return float(h), n, pos, neg


def deterministic_projection(hidden_dim: int, proj_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    return (rng.randn(int(hidden_dim), int(proj_dim)).astype(np.float32) / math.sqrt(float(proj_dim)))


def compute_basis(x: np.ndarray, k: int) -> np.ndarray:
    # Thin QR + small SVD gives the same left singular subspace as full SVD
    # but is much faster for sequence_len x projection_dim matrices.
    q, r = np.linalg.qr(np.asarray(x, dtype=np.float32), mode="reduced")
    ur, _s, _vh = np.linalg.svd(r.astype(np.float64), full_matrices=False)
    return (q @ ur[:, : int(k)]).astype(np.float64, copy=False)


def minor_signs(basis: np.ndarray, tuples: list[tuple[int, ...]]) -> np.ndarray:
    out = np.empty(len(tuples), dtype=np.float64)
    for i, tup in enumerate(tuples):
        sign, logabs = np.linalg.slogdet(basis[list(tup), :])
        # Sign entropy only needs the orientation sign. Keeping the sign from
        # slogdet avoids high-rank underflow when exp(logabs) rounds to zero.
        _ = logabs
        out[i] = float(sign)
    return out


def relation_token_sets(prompt_meta: dict, scrambled: bool) -> list[list[int]]:
    span_meta = prompt_meta.get("span_metadata") or {}
    relations = span_meta.get("relation_instances") or []
    item = prompt_meta.get("item") or {}
    arity = int(item.get("relation_arity") or 0)
    if not relations or arity <= 0:
        return []
    arg_matrix = [list(rel.get("argument_rep_tokens") or []) for rel in relations]
    if not arg_matrix or not all(len(row) == arity for row in arg_matrix):
        return []
    n = len(arg_matrix)
    token_sets = []
    for i in range(n):
        if scrambled:
            token_sets.append([arg_matrix[(i + role + 1) % n][role] for role in range(arity)])
        else:
            token_sets.append(list(arg_matrix[i]))
    return token_sets


def valid_tuple(vals: Iterable[int], k: int, seq_len: int) -> tuple[int, ...] | None:
    out = [int(x) for x in vals if x is not None and 0 <= int(x) < int(seq_len)]
    if len(out) != int(k) or len(set(out)) != len(out):
        return None
    return tuple(out)


def arity_tuples(prompt_meta: dict, k: int, scrambled: bool, seq_len: int) -> list[tuple[int, ...]]:
    out = []
    for toks in relation_token_sets(prompt_meta, scrambled=scrambled):
        if len(toks) < int(k):
            continue
        for combo in itertools.combinations(toks, int(k)):
            tup = valid_tuple(combo, int(k), seq_len)
            if tup is not None:
                out.append(tup)
    return out


def random_tuples(seq_len: int, k: int, n: int, seed: int, min_gap: int = 2, exclude=None) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(int(seed))
    excluded = set(exclude or [])
    out: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(10000, int(n) * 500)
    while len(out) < int(n) and attempts < max_attempts:
        attempts += 1
        cand = tuple(sorted(int(x) for x in rng.choice(seq_len, size=int(k), replace=False)))
        if cand in excluded or cand in out:
            continue
        if min_gap > 0 and any((b - a) < int(min_gap) for a, b in zip(cand, cand[1:])):
            continue
        out.add(cand)
    return sorted(out)


def bootstrap_ci(vals: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signflip_p(vals: np.ndarray, seed: int, n_perm: int = 10000) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    obs = float(vals.mean())
    rng = np.random.default_rng(int(seed))
    # Chunk to avoid allocating a huge matrix if this is run on a small pod.
    ge = 0
    done = 0
    chunk = 1000
    while done < int(n_perm):
        m = min(chunk, int(n_perm) - done)
        flips = rng.choice(np.array([-1.0, 1.0]), size=(m, vals.size), replace=True)
        ge += int(np.sum((flips * vals).mean(axis=1) >= obs))
        done += m
    return float((ge + 1) / (int(n_perm) + 1))


def bh_qvalues(pvals: list[float]) -> list[float]:
    p = np.asarray([1.0 if not np.isfinite(x) else float(x) for x in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.ones(n, dtype=float)
    for rank, idx in enumerate(order, start=1):
        q[idx] = p[idx] * n / rank
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return [float(x) for x in np.clip(q, 0.0, 1.0)]


def process_model(model_short: str, cfg: dict, proj_dims: list[int], seeds: list[int], out_dir: Path) -> None:
    cache = Path(cfg["cache"])
    arrays = cache / "arrays"
    prompts = cache / "prompts"
    prompt_rows = []
    summary_rows = []
    for arity in (3, 4, 5, 6):
        layer = int(cfg["layers"][arity])
        q_ids = sorted(
            int(p.stem.split("_")[-1])
            for p in arrays.glob("prompt_*.npz")
            if int(p.stem.split("_")[-1]) // 100 == (arity - 1) and int(p.stem.split("_")[-1]) % 2 == 1
        )
        vals_by_setting: dict[tuple[int, int], list[float]] = {
            (int(proj_dim), int(projection_seed)): [] for proj_dim in proj_dims for projection_seed in seeds
        }
        for q_pos, q_idx in enumerate(q_ids, start=1):
            z_path = arrays / f"prompt_{q_idx:04d}.npz"
            j_path = prompts / f"prompt_{q_idx:04d}.json"
            with np.load(z_path, allow_pickle=False) as z:
                hidden = z[f"hidden_L{layer}"].astype(np.float32, copy=False)
                seq_len = int(z["input_ids"].shape[0])
            with j_path.open() as f:
                meta = json.load(f)
            gt = arity_tuples(meta, arity, scrambled=False, seq_len=seq_len)
            scr = arity_tuples(meta, arity, scrambled=True, seq_len=seq_len)
            n = min(len(gt), len(scr), 20)
            gt = gt[:n]
            scr = scr[:n]
            rand_gt = random_tuples(seq_len, arity, n, seed=q_idx * 100000 + layer * 100 + arity + 1701, exclude=gt)
            rand_scr = random_tuples(seq_len, arity, n, seed=q_idx * 100000 + layer * 100 + arity + 2903, exclude=scr)
            n = min(n, len(rand_gt), len(rand_scr))
            gt, scr, rand_gt, rand_scr = gt[:n], scr[:n], rand_gt[:n], rand_scr[:n]
            for proj_dim in proj_dims:
                for projection_seed in seeds:
                    proj = deterministic_projection(hidden.shape[1], proj_dim, projection_seed)
                    basis = compute_basis(hidden @ proj, arity)
                    h_gt, n_gt, _pgt, _ngt = sign_entropy_from_signs(minor_signs(basis, gt))
                    h_scr, n_scr, _ps, _ns = sign_entropy_from_signs(minor_signs(basis, scr))
                    h_rgt, n_rgt, _prg, _nrg = sign_entropy_from_signs(minor_signs(basis, rand_gt))
                    h_rscr, n_rscr, _prs, _nrs = sign_entropy_from_signs(minor_signs(basis, rand_scr))
                    gap_gt = h_rgt - h_gt
                    gap_scr = h_rscr - h_scr
                    d_val = gap_gt - gap_scr
                    vals_by_setting[(int(proj_dim), int(projection_seed))].append(d_val)
                    prompt_rows.append({
                        "model": model_short,
                        "prompt_id": q_idx,
                        "relation_arity": arity,
                        "k": arity,
                        "layer": layer,
                        "projection_dim": proj_dim,
                        "projection_seed": projection_seed,
                        "tuple_count": n,
                        "H_ground_truth": h_gt,
                        "H_scrambled": h_scr,
                        "H_random_for_ground_truth": h_rgt,
                        "H_random_for_scrambled": h_rscr,
                        "gap_ground_truth": gap_gt,
                        "gap_scrambled": gap_scr,
                        "D_gt_minus_scrambled_gap": d_val,
                        "gt_valid_det_count": n_gt,
                        "scrambled_valid_det_count": n_scr,
                        "random_gt_valid_det_count": n_rgt,
                        "random_scrambled_valid_det_count": n_rscr,
                    })
            if q_pos % 10 == 0:
                print(f"[{model_short}] r={arity} loaded {q_pos}/{len(q_ids)} heldout prompts", flush=True)
        for proj_dim in proj_dims:
            for projection_seed in seeds:
                vals = vals_by_setting[(int(proj_dim), int(projection_seed))]
                arr = np.asarray(vals, dtype=float)
                ci_low, ci_high = bootstrap_ci(arr, seed=proj_dim * 1000 + projection_seed * 10 + arity)
                p = signflip_p(arr, seed=proj_dim * 2000 + projection_seed * 20 + arity)
                summary_rows.append({
                    "model": model_short,
                    "relation_arity": arity,
                    "k": arity,
                    "layer": layer,
                    "projection_dim": proj_dim,
                    "projection_seed": projection_seed,
                    "n_prompts": int(arr.size),
                    "mean_D": float(np.mean(arr)),
                    "std_D": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "positive_fraction": float(np.mean(arr > 0)),
                    "signflip_p": p,
                })
                print(f"[{model_short}] r={arity} p={proj_dim} seed={projection_seed} meanD={float(np.mean(arr)):+.4f}", flush=True)
    qvals = bh_qvalues([r["signflip_p"] for r in summary_rows])
    for row, q in zip(summary_rows, qvals):
        row["bh_q_value"] = q
        row["survives"] = bool(row["mean_D"] > 0 and row["ci_low"] > 0 and q < 0.05)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{model_short.lower()}_projection_dim_seed_prompt_rows.csv", prompt_rows)
    write_csv(out_dir / f"{model_short.lower()}_projection_dim_seed_summary.csv", summary_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(out_dir: Path) -> None:
    rows = []
    for p in out_dir.glob("*_projection_dim_seed_summary.csv"):
        with p.open() as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        for key in ("relation_arity", "k", "layer", "projection_dim", "projection_seed", "n_prompts"):
            r[key] = int(r[key])
        for key in ("mean_D", "std_D", "ci_low", "ci_high", "positive_fraction", "signflip_p", "bh_q_value"):
            r[key] = float(r[key])
        r["survives"] = str(r["survives"]).lower() == "true"
    write_csv(out_dir / "final_projection_dim_seed_matrix_summary.csv", rows)
    agg = []
    keys = sorted({(r["model"], r["relation_arity"], r["projection_dim"]) for r in rows})
    for model, arity, pdim in keys:
        sub = [r for r in rows if r["model"] == model and r["relation_arity"] == arity and r["projection_dim"] == pdim]
        means = np.asarray([r["mean_D"] for r in sub], dtype=float)
        agg.append({
            "model": model,
            "relation_arity": arity,
            "k": arity,
            "projection_dim": pdim,
            "n_projection_seeds": len(sub),
            "seed_mean_D": float(np.mean(means)),
            "seed_std_D": float(np.std(means, ddof=1)) if means.size > 1 else 0.0,
            "min_seed_mean_D": float(np.min(means)),
            "max_seed_mean_D": float(np.max(means)),
            "n_surviving_seeds": int(sum(r["survives"] for r in sub)),
            "all_seeds_positive": bool(np.all(means > 0)),
            "all_seeds_survive": bool(all(r["survives"] for r in sub)),
        })
    write_csv(out_dir / "final_projection_dim_seed_matrix_by_dim.csv", agg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="8B,70B,405B")
    ap.add_argument("--proj-dims", default="32,64,128")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out-dir", default="/workspace/projection_dim_seed_matrix_results")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    proj_dims = [int(x) for x in args.proj_dims.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    for model in [x for x in args.models.split(",") if x]:
        process_model(model, MODELS[model], proj_dims, seeds, out_dir)
    aggregate(out_dir)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
