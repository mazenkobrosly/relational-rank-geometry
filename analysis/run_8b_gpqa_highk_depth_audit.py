#!/usr/bin/env python3
"""8B GPQA high-k accessible-rank depth audit.

This is intentionally separate from the semantic-rank ladder. It asks whether
the original GPQA-198 setting shows higher-k Pluecker sign structure when we
scan a broad layer profile and audit tuple budgets / determinant margins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
    bh_qvalues,
    compute_basis,
    compute_minors,
    deterministic_projection,
    iter_cached_payloads,
    parse_int_list,
)
from analysis.run_semantic_rank_phase2 import (  # noqa: E402
    bidir_no_self,
    candidate_tuple_array,
    pair_matrix_values,
    position_matched_from_pool,
    norm_matched_from_pool,
    random_tuples_from_pool,
    select_by_degree,
    token_meta,
    top_scored_tuples,
)


SELECTOR_LABELS = {
    "degree_salience": "degree/salience",
    "avg_pair_attention": "average pairwise mutual attention",
    "k_clique_all_pairs_attention": "k-clique / geometric all-pairs attention",
    "random_control": "random control",
    "position_matched_control": "position-matched control",
    "norm_matched_control": "norm-matched control",
}

PRIMARY_SELECTORS = {
    "degree_salience",
    "avg_pair_attention",
    "k_clique_all_pairs_attention",
}


def parse_float_list(raw: str | None, default: Sequence[float]) -> List[float]:
    if raw is None or not str(raw).strip():
        return list(default)
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def entropy_from_signed_minors(minors: Sequence[float], eps: float = 0.0) -> Tuple[float, int]:
    vals = np.asarray([float(m) for m in minors if np.isfinite(m) and abs(float(m)) > float(eps)], dtype=float)
    if vals.size < 2:
        return 0.0, int(vals.size)
    p = float(np.mean(vals > 0))
    if p <= 0.0 or p >= 1.0:
        return 0.0, int(vals.size)
    h = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    return float(h), int(vals.size)


def absdet_stats(minors: Sequence[float], prefix: str) -> Dict[str, float]:
    vals = np.asarray([abs(float(m)) for m in minors if np.isfinite(m)], dtype=float)
    if vals.size == 0:
        return {
            f"{prefix}_abs_det_mean": np.nan,
            f"{prefix}_abs_det_median": np.nan,
            f"{prefix}_abs_det_q05": np.nan,
            f"{prefix}_abs_det_q10": np.nan,
            f"{prefix}_abs_det_q20": np.nan,
            f"{prefix}_abs_det_min": np.nan,
        }
    return {
        f"{prefix}_abs_det_mean": float(np.mean(vals)),
        f"{prefix}_abs_det_median": float(np.median(vals)),
        f"{prefix}_abs_det_q05": float(np.quantile(vals, 0.05)),
        f"{prefix}_abs_det_q10": float(np.quantile(vals, 0.10)),
        f"{prefix}_abs_det_q20": float(np.quantile(vals, 0.20)),
        f"{prefix}_abs_det_min": float(np.min(vals)),
    }


def top_by_scores(cands: np.ndarray, scores: np.ndarray, n: int) -> List[Tuple[int, ...]]:
    return top_scored_tuples(cands, scores, n)


def build_selector_prefixes(
    *,
    attn: np.ndarray,
    norms: np.ndarray,
    k: int,
    max_budget: int,
    n_candidates: int,
    n_hub_tokens: int,
    min_gap: int,
    position_bins: int,
    norm_bins: int,
    seed_base: int,
) -> Dict[str, List[Tuple[int, ...]]]:
    seq_len = int(attn.shape[0])
    degree = np.asarray(attn, dtype=float).sum(axis=1)
    raw_hubs = select_by_degree(attn, None, n_hub_tokens, min_gap)
    all_pool = list(range(seq_len))
    if len(raw_hubs) < k:
        return {name: [] for name in SELECTOR_LABELS}

    # Generate one candidate universe per prompt/layer/k. Budgets are prefixes
    # of the ranked list, so the 200/500/1000 audit is nested and comparable.
    cands = candidate_tuple_array(
        raw_hubs,
        k,
        int(max_budget),
        max(int(n_candidates), int(max_budget) * 20),
        int(seed_base) + 100,
        int(min_gap),
    )
    if cands.size == 0:
        return {name: [] for name in SELECTOR_LABELS}

    pair_vals = pair_matrix_values(attn, cands)
    avg_pair_scores = pair_vals.mean(axis=1) if pair_vals.size else np.zeros(len(cands), dtype=float)
    clique_scores = np.exp(np.log(np.maximum(pair_vals, 1e-12)).mean(axis=1)) if pair_vals.size else avg_pair_scores
    degree_scores = degree[cands].mean(axis=1)

    avg_tuples = top_by_scores(cands, avg_pair_scores, max_budget)
    degree_tuples = top_by_scores(cands, degree_scores, max_budget)
    clique_tuples = top_by_scores(cands, clique_scores, max_budget)
    random_tuples = random_tuples_from_pool(
        all_pool,
        k,
        max_budget,
        int(seed_base) + 200,
        min_gap,
        exclude=avg_tuples,
    )
    pos_tuples = position_matched_from_pool(
        avg_tuples,
        all_pool,
        seq_len,
        int(seed_base) + 300,
        position_bins,
        min_gap,
    )[:max_budget]
    norm_tuples = norm_matched_from_pool(
        avg_tuples,
        all_pool,
        norms,
        int(seed_base) + 400,
        norm_bins,
        min_gap,
    )[:max_budget]

    out = {
        "degree_salience": degree_tuples,
        "avg_pair_attention": avg_tuples,
        "k_clique_all_pairs_attention": clique_tuples if k >= 4 else [],
        "random_control": random_tuples,
        "position_matched_control": pos_tuples,
        "norm_matched_control": norm_tuples,
    }
    return out


def process_prompt(job: Tuple) -> List[Dict[str, object]]:
    (
        cache_dir,
        prompt_id,
        layers,
        ranks,
        low_k_budget,
        high_k_budgets,
        det_trim_quantiles,
        n_hub_tokens,
        n_candidates,
        min_gap,
        position_bins,
        norm_bins,
        proj_dim,
        projection_seed,
    ) = job
    payload = next(iter_cached_payloads(cache_dir, layers=layers, prompt_ids=[int(prompt_id)]))
    rows: List[Dict[str, object]] = []
    meta = token_meta(payload)
    q_idx = int(payload["q_idx"])
    item = payload.get("item") or {}
    seq_len = int(payload.get("seq_len") or 0)

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
            budgets = [int(low_k_budget)] if int(k) < 4 else [int(x) for x in high_k_budgets]
            max_budget = max(budgets)
            seed_base = q_idx * 100000 + int(layer) * 100 + int(k)
            selector_prefixes = build_selector_prefixes(
                attn=attn,
                norms=norms,
                k=int(k),
                max_budget=max_budget,
                n_candidates=int(n_candidates),
                n_hub_tokens=int(n_hub_tokens),
                min_gap=int(min_gap),
                position_bins=int(position_bins),
                norm_bins=int(norm_bins),
                seed_base=seed_base,
            )
            random_prefix = selector_prefixes.get("random_control", [])
            for budget in budgets:
                random_tuples = random_prefix[:budget]
                random_minors = compute_minors(basis, random_tuples)
                for selector, tuples_full in selector_prefixes.items():
                    if selector == "k_clique_all_pairs_attention" and int(k) < 4:
                        continue
                    tuples = tuples_full[:budget]
                    if not tuples:
                        continue
                    selected_minors = random_minors if selector == "random_control" else compute_minors(basis, tuples)
                    trims = [0.0]
                    if int(k) in {4, 5, 6}:
                        trims = list(det_trim_quantiles)
                    pooled_abs = np.asarray(
                        [abs(float(x)) for x in list(selected_minors) + list(random_minors) if np.isfinite(x)],
                        dtype=float,
                    )
                    for trim in trims:
                        threshold = float(np.quantile(pooled_abs, float(trim))) if pooled_abs.size and float(trim) > 0 else 0.0
                        h_sel, n_sel_after = entropy_from_signed_minors(selected_minors, eps=threshold)
                        h_rand, n_rand_after = entropy_from_signed_minors(random_minors, eps=threshold)
                        rows.append(
                            {
                                "model": "meta-llama/Llama-3.1-8B-Instruct",
                                "benchmark": "GPQA-198",
                                "prompt_id": q_idx,
                                "qid": item.get("qid", q_idx),
                                "layer": int(layer),
                                "k": int(k),
                                "tuple_budget": int(budget),
                                "selector": selector,
                                "selector_label": SELECTOR_LABELS.get(selector, selector),
                                "det_trim_quantile": float(trim),
                                "det_threshold_abs": threshold,
                                "H_selector": h_sel,
                                "H_random": h_rand,
                                "random_minus_selector_entropy_gap": float(h_rand - h_sel),
                                "n_selector_tuples": int(len(selected_minors)),
                                "n_random_tuples": int(len(random_minors)),
                                "n_selector_after_trim": int(n_sel_after),
                                "n_random_after_trim": int(n_rand_after),
                                "seq_len": seq_len,
                                "pred_letter": payload.get("pred_letter"),
                                "correct_letter": payload.get("correct_letter"),
                                "is_correct": payload.get("is_correct"),
                                **absdet_stats(selected_minors, "selector"),
                                **absdet_stats(random_minors, "random"),
                            }
                        )
    return rows


def prompt_ids_from_cache(cache_dir: Path) -> List[int]:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = manifest.get("prompt_ids") or manifest.get("requested_prompt_ids") or []
        if ids:
            return sorted(int(x) for x in ids)
    return sorted(int(p.stem.split("_")[-1]) for p in (cache_dir / "arrays").glob("prompt_*.npz"))


def bootstrap_ci(vals: Sequence[float], seed: int = 20260508, n_boot: int = 1000) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(rng.choice(arr, size=arr.size, replace=True)) for _ in range(int(n_boot))], dtype=float)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize(prompt_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["layer", "k", "tuple_budget", "selector", "selector_label", "det_trim_quantile"]
    rows = []
    for keys, g in prompt_df.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g["random_minus_selector_entropy_gap"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        h_sel = pd.to_numeric(g["H_selector"], errors="coerce").to_numpy(dtype=float)
        h_rand = pd.to_numeric(g["H_random"], errors="coerce").to_numpy(dtype=float)
        if vals.size > 1 and float(np.std(vals, ddof=1)) > 0.0:
            t_stat, p_value = stats.ttest_1samp(vals, popmean=0.0, alternative="greater")
            t_stat = float(t_stat)
            p_value = float(p_value)
        else:
            t_stat = np.nan
            p_value = 1.0
        ci_lo, ci_hi = bootstrap_ci(vals)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_prompts": int(vals.size),
                "mean_H_selector": float(np.nanmean(h_sel)),
                "mean_H_random": float(np.nanmean(h_rand)),
                "mean_random_minus_selector_entropy_gap": float(np.nanmean(vals)) if vals.size else np.nan,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "paired_prompt_t_stat": t_stat,
                "paired_prompt_p_greater": p_value,
                "positive_prompt_fraction": float(np.mean(vals > 0)) if vals.size else np.nan,
                "mean_selector_abs_det": float(np.nanmean(pd.to_numeric(g["selector_abs_det_mean"], errors="coerce"))),
                "mean_random_abs_det": float(np.nanmean(pd.to_numeric(g["random_abs_det_mean"], errors="coerce"))),
                "mean_n_selector_after_trim": float(np.nanmean(pd.to_numeric(g["n_selector_after_trim"], errors="coerce"))),
                "mean_n_random_after_trim": float(np.nanmean(pd.to_numeric(g["n_random_after_trim"], errors="coerce"))),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q_value"] = bh_qvalues(out["paired_prompt_p_greater"].to_numpy(dtype=float))
    return out


def preferred_budget_for_k(k: int, high_k_budgets: Sequence[int], low_k_budget: int) -> int:
    return int(low_k_budget) if int(k) < 4 else int(max(high_k_budgets))


def best_primary_table(stats_df: pd.DataFrame, high_k_budgets: Sequence[int], low_k_budget: int) -> pd.DataFrame:
    rows = []
    for (layer, k), g in stats_df.groupby(["layer", "k"]):
        pref_budget = preferred_budget_for_k(int(k), high_k_budgets, low_k_budget)
        sub = g[
            (g["tuple_budget"] == pref_budget)
            & (g["det_trim_quantile"] == 0.0)
            & (g["selector"].isin(PRIMARY_SELECTORS))
        ].copy()
        if sub.empty:
            continue
        best = sub.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).iloc[0].to_dict()
        rows.append(best)
    return pd.DataFrame(rows)


def plot_depth_heatmap(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return
    piv = best_df.pivot_table(
        index="k",
        columns="layer",
        values="mean_random_minus_selector_entropy_gap",
        aggfunc="max",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(x) for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"k={x}" for x in piv.index])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank")
    ax.set_title("8B GPQA High-k Depth Audit: best primary selector gap")
    for i, k in enumerate(piv.index):
        for j, layer in enumerate(piv.columns):
            val = piv.loc[k, layer]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > np.nanmax(piv.to_numpy()) * 0.45 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="random - selector entropy gap")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_kmax_by_layer(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return
    layer_best = best_df.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).groupby("layer", as_index=False).first()
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(layer_best["layer"].astype(str), layer_best["k"], color="#3b82f6", alpha=0.75)
    ax1.set_ylabel("rank with max gap")
    ax1.set_xlabel("Layer")
    ax1.set_title("8B GPQA accessible-rank peak by layer")
    ax1.set_ylim(0, max(8, int(layer_best["k"].max()) + 1))
    ax2 = ax1.twinx()
    ax2.plot(layer_best["layer"].astype(str), layer_best["mean_random_minus_selector_entropy_gap"], color="#f59e0b", marker="o", linewidth=2)
    ax2.set_ylabel("max entropy gap")
    for x, (_, row) in enumerate(layer_best.iterrows()):
        ax1.text(x, row["k"] + 0.1, f"k={int(row['k'])}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_budget_audit(stats_df: pd.DataFrame, out_path: Path) -> None:
    sub = stats_df[
        (stats_df["k"] >= 4)
        & (stats_df["det_trim_quantile"] == 0.0)
        & (stats_df["selector"].isin(PRIMARY_SELECTORS))
    ].copy()
    if sub.empty:
        return
    # Best over layer/selector per k and budget.
    best = sub.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).groupby(["k", "tuple_budget"], as_index=False).first()
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, g in best.groupby("k"):
        g = g.sort_values("tuple_budget")
        ax.plot(g["tuple_budget"], g["mean_random_minus_selector_entropy_gap"], marker="o", linewidth=2, label=f"k={int(k)}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(best["tuple_budget"].unique()))
    ax.set_xticklabels([str(int(x)) for x in sorted(best["tuple_budget"].unique())])
    ax.set_xlabel("Tuple budget")
    ax.set_ylabel("best primary-selector entropy gap")
    ax.set_title("8B GPQA high-k tuple-budget audit")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary(stats_df: pd.DataFrame, best_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# 8B GPQA High-k Depth Audit Summary",
        "",
        "This audit is separate from the semantic-rank ladder. It uses GPQA-198 and a broad layer scan to test whether high-k structure appears outside the late-layer subset used in the semantic ladder run.",
        "",
        "## Main Peak Table",
        "",
    ]
    if best_df.empty:
        lines.append("No best-primary rows were produced.")
    else:
        top = best_df.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).head(20)
        for _, r in top.iterrows():
            lines.append(
                f"- layer {int(r['layer'])}, k={int(r['k'])}, budget={int(r['tuple_budget'])}, "
                f"selector={r['selector_label']}: gap={float(r['mean_random_minus_selector_entropy_gap']):+.4f}, "
                f"CI=[{float(r['bootstrap_ci_low']):+.4f},{float(r['bootstrap_ci_high']):+.4f}], "
                f"q={float(r['fdr_q_value']):.3g}, positive={float(r['positive_prompt_fraction']):.3f}"
            )
    lines.extend(["", "## Best Layer Per Rank", ""])
    if not best_df.empty:
        rank_best = best_df.sort_values("mean_random_minus_selector_entropy_gap", ascending=False).groupby("k", as_index=False).first()
        for _, r in rank_best.iterrows():
            lines.append(
                f"- k={int(r['k'])}: best layer {int(r['layer'])}, selector={r['selector_label']}, "
                f"budget={int(r['tuple_budget'])}, gap={float(r['mean_random_minus_selector_entropy_gap']):+.4f}, "
                f"q={float(r['fdr_q_value']):.3g}"
            )
    lines.extend(
        [
            "",
            "## Determinant-Margin Note",
            "",
            "For k=4,5,6 the CSV includes det_trim_quantile in {0, .05, .10, .20}. These rows recompute sign entropy after removing near-zero determinants using a pooled selector+random absolute-determinant threshold per prompt/layer/k/selector/budget.",
            "",
            "## Interpretation Guardrail",
            "",
            "Do not use the semantic ladder alone to reject high-k structure. This GPQA broad-depth audit is the relevant check for the original 8B accessible-rank claim.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-cache-dir", required=True)
    p.add_argument("--layers", default="0,5,10,15,20,25,30,31")
    p.add_argument("--ranks", default="2,3,4,5,6,7")
    p.add_argument("--low-k-budget", type=int, default=200)
    p.add_argument("--high-k-budgets", default="200,500,1000")
    p.add_argument("--det-trim-quantiles", default="0,0.05,0.10,0.20")
    p.add_argument("--n-hub-tokens", type=int, default=60)
    p.add_argument("--n-candidates", type=int, default=20000)
    p.add_argument("--min-gap", type=int, default=2)
    p.add_argument("--position-bins", type=int, default=10)
    p.add_argument("--norm-bins", type=int, default=10)
    p.add_argument("--projection-seed", type=int, default=42)
    p.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    p.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4) - 1)))
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.input_cache_dir).expanduser().resolve()
    layers = parse_int_list(args.layers, [])
    ranks = parse_int_list(args.ranks, [])
    high_k_budgets = parse_int_list(args.high_k_budgets, [200, 500, 1000])
    det_trim_quantiles = parse_float_list(args.det_trim_quantiles, [0.0, 0.05, 0.10, 0.20])
    results_dir = Path(args.results_dir).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = prompt_ids_from_cache(cache_dir)
    if args.max_prompts is not None:
        prompt_ids = prompt_ids[: int(args.max_prompts)]
    print(f"cache={cache_dir}", flush=True)
    print(f"prompts={len(prompt_ids)} layers={layers} ranks={ranks} high_k_budgets={high_k_budgets}", flush=True)

    jobs = [
        (
            str(cache_dir),
            int(prompt_id),
            list(layers),
            list(ranks),
            int(args.low_k_budget),
            list(high_k_budgets),
            list(det_trim_quantiles),
            int(args.n_hub_tokens),
            int(args.n_candidates),
            int(args.min_gap),
            int(args.position_bins),
            int(args.norm_bins),
            int(args.proj_dim),
            int(args.projection_seed),
        )
        for prompt_id in prompt_ids
    ]
    all_rows: List[Dict[str, object]] = []
    if int(args.workers) <= 1:
        for pos, job in enumerate(jobs, start=1):
            rows = process_prompt(job)
            all_rows.extend(rows)
            if pos % max(1, int(args.progress_every)) == 0:
                print(f"[live] high-k audit {pos}/{len(jobs)} prompts; rows={len(all_rows)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(process_prompt, job) for job in jobs]
            for pos, fut in enumerate(as_completed(futs), start=1):
                rows = fut.result()
                all_rows.extend(rows)
                if pos % max(1, int(args.progress_every)) == 0:
                    print(f"[live] high-k audit {pos}/{len(jobs)} prompts; rows={len(all_rows)}", flush=True)

    prompt_df = pd.DataFrame(all_rows)
    prompt_rows_path = results_dir / "8b_gpqa_highk_depth_audit_prompt_rows.csv"
    prompt_df.to_csv(prompt_rows_path, index=False)
    print(f"Wrote {prompt_rows_path}", flush=True)

    stats_df = summarize(prompt_df)
    stats_path = results_dir / "8b_gpqa_highk_depth_audit.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path}", flush=True)

    det_path = results_dir / "8b_gpqa_highk_det_margin_audit.csv"
    stats_df[stats_df["det_trim_quantile"] > 0].to_csv(det_path, index=False)
    print(f"Wrote {det_path}", flush=True)

    best_df = best_primary_table(stats_df, high_k_budgets, int(args.low_k_budget))
    best_path = results_dir / "8b_gpqa_highk_best_primary_by_layer_rank.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Wrote {best_path}", flush=True)

    plot_depth_heatmap(best_df, figures_dir / "fig_8b_highk_depth_heatmap.png")
    plot_kmax_by_layer(best_df, figures_dir / "fig_8b_kmax_by_layer.png")
    plot_budget_audit(stats_df, figures_dir / "fig_8b_highk_tuple_budget_audit.png")
    print("Wrote figures", flush=True)

    write_summary(stats_df, best_df, results_dir / "8b_highk_audit_summary.md")
    print(f"Wrote {results_dir / '8b_highk_audit_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
