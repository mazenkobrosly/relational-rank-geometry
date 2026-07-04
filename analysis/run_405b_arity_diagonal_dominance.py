#!/usr/bin/env python3
"""405B relational-arity diagonal-dominance audit.

This recomputes off-diagonal argument tuple sets from the saved arity cache.
The original arity rows only contain ground-truth argument tuples at k=r and
argument+predicate tuples at k=r+1, so the full arity x rank heatmap cannot be
honestly computed from those rows alone.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.plucker_controls import (  # noqa: E402
    DEFAULT_PROJ_DIM,
    compute_basis,
    compute_minors,
    deterministic_projection,
    find_random_tuples,
    iter_cached_payloads,
    parse_int_list,
)
from analysis.run_405b_relational_arity_benchmark import (  # noqa: E402
    DEFAULT_LAYERS,
    DEFAULT_RANKS,
    entropy_from_minors,
)


ARITIES = [3, 4, 5, 6]
MODES = {
    "argument_subsets": "ground-truth argument subsets",
    "argument_plus_predicate_subsets": "predicate + argument subsets",
}


def bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
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


def bootstrap_ci(vals: Sequence[float], seed: int = 20260508, n_boot: int = 2000) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(rng.choice(arr, size=arr.size, replace=True)) for _ in range(int(n_boot))])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def valid_tuple(vals: Sequence[int | None], k: int, seq_len: int) -> Tuple[int, ...] | None:
    out = [int(x) for x in vals if x is not None and 0 <= int(x) < int(seq_len)]
    if len(out) != int(k) or len(set(out)) != len(out):
        return None
    return tuple(out)


def relation_token_sets(payload: Mapping, mode: str, scrambled: bool) -> List[List[int]]:
    span_meta = payload.get("span_metadata") or {}
    relations = span_meta.get("relation_instances") or []
    item = payload.get("item") or {}
    arity = int(item.get("relation_arity") or 0)
    if not relations or arity <= 0:
        return []

    arg_matrix = [list(rel.get("argument_rep_tokens") or []) for rel in relations]
    if not arg_matrix or not all(len(row) == arity for row in arg_matrix):
        return []
    n = len(arg_matrix)

    token_sets: List[List[int]] = []
    for i, rel in enumerate(relations):
        if scrambled:
            args = [arg_matrix[(i + role + 1) % n][role] for role in range(arity)]
        else:
            args = list(arg_matrix[i])
        if mode == "argument_subsets":
            token_sets.append(args)
        elif mode == "argument_plus_predicate_subsets":
            token_sets.append([rel.get("predicate_rep_token")] + args)
        else:
            raise ValueError(f"unknown mode: {mode}")
    return token_sets


def tuple_subsets(payload: Mapping, mode: str, k: int, scrambled: bool) -> List[Tuple[int, ...]]:
    seq_len = int(payload.get("seq_len") or 0)
    out: List[Tuple[int, ...]] = []
    for toks in relation_token_sets(payload, mode, scrambled=scrambled):
        if int(k) > len(toks):
            continue
        for combo in itertools.combinations(toks, int(k)):
            tup = valid_tuple(combo, int(k), seq_len)
            if tup is not None:
                out.append(tup)
    return out


def process_prompt(job: Tuple) -> List[Dict[str, object]]:
    cache_dir, q_idx, layers, ranks, proj_dim, projection_seed, min_gap = job
    payload = next(iter_cached_payloads(Path(cache_dir), layers=layers, prompt_ids=[int(q_idx)]))
    item = payload.get("item") or {}
    arity = int(item.get("relation_arity") or 0)
    if arity not in ARITIES:
        return []
    rows: List[Dict[str, object]] = []
    seq_len = int(payload.get("seq_len") or 0)
    max_rank = int(max(ranks))

    tuple_cache: Dict[Tuple[str, int], Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]] = {}
    for mode in MODES:
        for k in ranks:
            gt_tuples = tuple_subsets(payload, mode, int(k), scrambled=False)
            scrambled_tuples = tuple_subsets(payload, mode, int(k), scrambled=True)
            if not gt_tuples or not scrambled_tuples:
                continue
            n = min(len(gt_tuples), len(scrambled_tuples))
            tuple_cache[(mode, int(k))] = (gt_tuples[:n], scrambled_tuples[:n])
    if not tuple_cache:
        return []

    projection = None

    for layer in layers:
        if layer not in payload["hidden_by_layer"]:
            continue
        hidden = payload["hidden_by_layer"][layer].astype(np.float32, copy=False)
        if projection is None:
            projection = deterministic_projection(hidden.shape[1], int(proj_dim), int(projection_seed))
        x_proj = hidden @ projection
        basis_full = compute_basis(x_proj, max_rank, center=False)
        for k in ranks:
            basis = basis_full[:, : int(k)]
            for mode in MODES:
                cached = tuple_cache.get((mode, int(k)))
                if cached is None:
                    continue
                gt_tuples, scrambled_tuples = cached
                n = len(gt_tuples)
                seed_base = int(q_idx) * 100000 + int(layer) * 100 + int(k)
                random_gt = find_random_tuples(
                    seq_len, int(k), n, seed_base + 1701, int(min_gap), exclude=gt_tuples
                )
                random_scrambled = find_random_tuples(
                    seq_len, int(k), n, seed_base + 2903, int(min_gap), exclude=scrambled_tuples
                )

                gt_minors = compute_minors(basis, gt_tuples)
                scrambled_minors = compute_minors(basis, scrambled_tuples)
                random_gt_minors = compute_minors(basis, random_gt)
                random_scrambled_minors = compute_minors(basis, random_scrambled)
                h_gt, n_gt = entropy_from_minors(gt_minors)
                h_scr, n_scr = entropy_from_minors(scrambled_minors)
                h_rgt, n_rgt = entropy_from_minors(random_gt_minors)
                h_rscr, n_rscr = entropy_from_minors(random_scrambled_minors)
                gt_gap = float(h_rgt - h_gt)
                scrambled_gap = float(h_rscr - h_scr)

                rows.append(
                    {
                        "row_type": "prompt_cell",
                        "prompt_id": int(q_idx),
                        "task_prompt_id": item.get("prompt_id"),
                        "relation_arity": arity,
                        "layer": int(layer),
                        "k": int(k),
                        "mode": mode,
                        "mode_label": MODES[mode],
                        "expected_k": int(arity if mode == "argument_subsets" else arity + 1),
                        "tuple_count": int(n),
                        "gt_valid_det_count": int(n_gt),
                        "scrambled_valid_det_count": int(n_scr),
                        "random_gt_valid_det_count": int(n_rgt),
                        "random_scrambled_valid_det_count": int(n_rscr),
                        "H_ground_truth": h_gt,
                        "H_scrambled": h_scr,
                        "H_random_for_ground_truth": h_rgt,
                        "H_random_for_scrambled": h_rscr,
                        "gap_ground_truth": gt_gap,
                        "gap_scrambled": scrambled_gap,
                        "D_gt_minus_scrambled_gap": float(gt_gap - scrambled_gap),
                    }
                )
    return rows


def prompt_ids_from_cache(cache_dir: Path) -> List[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in (cache_dir / "arrays").glob("prompt_*.npz"))


def summarize_cells(prompt_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, g in prompt_df.groupby(["mode", "relation_arity", "k", "layer"], dropna=False):
        mode, arity, k, layer = keys
        vals = pd.to_numeric(g["D_gt_minus_scrambled_gap"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 1 and float(np.std(vals, ddof=1)) > 0:
            t_stat, p_value = stats.ttest_1samp(vals, 0.0, alternative="greater")
        else:
            t_stat, p_value = np.nan, 1.0
        ci_low, ci_high = bootstrap_ci(vals, seed=1000 + int(arity) * 100 + int(k) * 10 + int(layer))
        rows.append(
            {
                "row_type": "cell",
                "mode": mode,
                "mode_label": MODES.get(mode, mode),
                "relation_arity": int(arity),
                "k": int(k),
                "layer": int(layer),
                "expected_k": int(arity if mode == "argument_subsets" else arity + 1),
                "n_prompts": int(vals.size),
                "mean_D_gt_minus_scrambled_gap": float(np.mean(vals)) if vals.size else np.nan,
                "D_ci_low": ci_low,
                "D_ci_high": ci_high,
                "D_positive_fraction": float(np.mean(vals > 0)) if vals.size else np.nan,
                "D_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "D_p_greater": float(p_value),
                "mean_tuple_count": float(pd.to_numeric(g["tuple_count"], errors="coerce").mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["D_q_value"] = bh_qvalues(out["D_p_greater"].to_numpy(dtype=float))
    return out


def best_cell_table(cell_df: pd.DataFrame) -> pd.DataFrame:
    if cell_df.empty:
        return cell_df.copy()
    idx = cell_df.groupby(["mode", "relation_arity", "k"])["mean_D_gt_minus_scrambled_gap"].idxmax()
    out = cell_df.loc[idx].copy()
    out["layer_policy"] = "best_layer_per_cell"
    return out


def paired_margin(prompt_df: pd.DataFrame, diag: pd.Series, off: pd.Series) -> Dict[str, object]:
    d = prompt_df[
        (prompt_df["mode"] == diag["mode"])
        & (prompt_df["relation_arity"] == int(diag["relation_arity"]))
        & (prompt_df["k"] == int(diag["k"]))
        & (prompt_df["layer"] == int(diag["layer"]))
    ][["prompt_id", "D_gt_minus_scrambled_gap"]].rename(columns={"D_gt_minus_scrambled_gap": "diag_D"})
    o = prompt_df[
        (prompt_df["mode"] == off["mode"])
        & (prompt_df["relation_arity"] == int(off["relation_arity"]))
        & (prompt_df["k"] == int(off["k"]))
        & (prompt_df["layer"] == int(off["layer"]))
    ][["prompt_id", "D_gt_minus_scrambled_gap"]].rename(columns={"D_gt_minus_scrambled_gap": "off_D"})
    merged = d.merge(o, on="prompt_id", how="inner")
    vals = (merged["diag_D"] - merged["off_D"]).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size > 1 and float(np.std(vals, ddof=1)) > 0:
        t_stat, p_value = stats.ttest_1samp(vals, 0.0, alternative="greater")
    else:
        t_stat, p_value = np.nan, 1.0
    ci_low, ci_high = bootstrap_ci(vals, seed=7000 + int(diag["relation_arity"]) * 100 + int(diag["k"]))
    return {
        "n_prompts": int(vals.size),
        "diagonal_margin": float(np.mean(vals)) if vals.size else np.nan,
        "margin_ci_low": ci_low,
        "margin_ci_high": ci_high,
        "margin_positive_fraction": float(np.mean(vals > 0)) if vals.size else np.nan,
        "margin_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "margin_p_greater": float(p_value),
    }


def dominance_tests(prompt_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (mode, arity), g in best_df.groupby(["mode", "relation_arity"], dropna=False):
        expected_k = int(arity if mode == "argument_subsets" else arity + 1)
        diag_rows = g[g["k"] == expected_k]
        if diag_rows.empty:
            continue
        diag = diag_rows.sort_values("mean_D_gt_minus_scrambled_gap", ascending=False).iloc[0]
        groups = {
            "lower_k_than_expected": g[g["k"] < expected_k],
            "higher_k_than_expected": g[g["k"] > expected_k],
            "max_off_diagonal": g[g["k"] != expected_k],
        }
        for group_name, off_g in groups.items():
            if off_g.empty:
                rows.append(
                    {
                        "row_type": "row_dominance",
                        "mode": mode,
                        "mode_label": MODES.get(mode, mode),
                        "relation_arity": int(arity),
                        "expected_k": expected_k,
                        "comparison_group": group_name,
                        "diagonal_k": int(diag["k"]),
                        "diagonal_layer": int(diag["layer"]),
                        "diagonal_D": float(diag["mean_D_gt_minus_scrambled_gap"]),
                        "offdiag_k": np.nan,
                        "offdiag_layer": np.nan,
                        "max_offdiag_D": np.nan,
                        "diagonal_margin": np.nan,
                        "dominance_supported": np.nan,
                    }
                )
                continue
            off = off_g.sort_values("mean_D_gt_minus_scrambled_gap", ascending=False).iloc[0]
            margin = paired_margin(prompt_df, diag, off)
            rows.append(
                {
                    "row_type": "row_dominance",
                    "mode": mode,
                    "mode_label": MODES.get(mode, mode),
                    "relation_arity": int(arity),
                    "expected_k": expected_k,
                    "comparison_group": group_name,
                    "diagonal_k": int(diag["k"]),
                    "diagonal_layer": int(diag["layer"]),
                    "diagonal_D": float(diag["mean_D_gt_minus_scrambled_gap"]),
                    "offdiag_k": int(off["k"]),
                    "offdiag_layer": int(off["layer"]),
                    "max_offdiag_D": float(off["mean_D_gt_minus_scrambled_gap"]),
                    **margin,
                    "dominance_supported": bool(margin["diagonal_margin"] > 0 and margin["margin_p_greater"] < 0.05),
                }
            )

    for mode, g_mode in best_df.groupby("mode", dropna=False):
        for k, gk in g_mode.groupby("k", dropna=False):
            expected_arity = int(k if mode == "argument_subsets" else k - 1)
            if expected_arity not in ARITIES:
                continue
            diag_rows = gk[gk["relation_arity"] == expected_arity]
            off_g = gk[gk["relation_arity"] != expected_arity]
            if diag_rows.empty or off_g.empty:
                continue
            diag = diag_rows.sort_values("mean_D_gt_minus_scrambled_gap", ascending=False).iloc[0]
            off = off_g.sort_values("mean_D_gt_minus_scrambled_gap", ascending=False).iloc[0]
            margin = paired_margin(prompt_df, diag, off)
            rows.append(
                {
                    "row_type": "column_dominance",
                    "mode": mode,
                    "mode_label": MODES.get(mode, mode),
                    "relation_arity": int(expected_arity),
                    "expected_k": int(k),
                    "comparison_group": "same_k_other_arities",
                    "diagonal_k": int(diag["k"]),
                    "diagonal_layer": int(diag["layer"]),
                    "diagonal_D": float(diag["mean_D_gt_minus_scrambled_gap"]),
                    "offdiag_arity": int(off["relation_arity"]),
                    "offdiag_k": int(off["k"]),
                    "offdiag_layer": int(off["layer"]),
                    "max_offdiag_D": float(off["mean_D_gt_minus_scrambled_gap"]),
                    **margin,
                    "dominance_supported": bool(margin["diagonal_margin"] > 0 and margin["margin_p_greater"] < 0.05),
                }
            )
    out = pd.DataFrame(rows)
    if "margin_p_greater" in out.columns:
        mask = out["margin_p_greater"].notna()
        out.loc[mask, "margin_q_value"] = bh_qvalues(out.loc[mask, "margin_p_greater"].to_numpy(dtype=float))
    return out


def plot_heatmap(best_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        sub = best_df[best_df["mode"] == mode]
        mat = sub.pivot(index="relation_arity", columns="k", values="mean_D_gt_minus_scrambled_gap").reindex(ARITIES)
        mat = mat.reindex(columns=list(range(1, 8)))
        im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-0.2, vmax=0.7)
        ax.set_title(MODES[mode])
        ax.set_xlabel("rank k")
        ax.set_ylabel("relation arity r")
        ax.set_xticks(range(7), labels=list(range(1, 8)))
        ax.set_yticks(range(len(ARITIES)), labels=ARITIES)
        for i, arity in enumerate(ARITIES):
            expected = arity if mode == "argument_subsets" else arity + 1
            if 1 <= expected <= 7:
                ax.scatter([expected - 1], [i], marker="s", s=150, facecolors="none", edgecolors="black", linewidths=1.5)
            for j, k in enumerate(range(1, 8)):
                val = mat.loc[arity, k] if k in mat.columns else np.nan
                label = "" if pd.isna(val) else f"{val:+.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("D = gap(true relation tuple) - gap(scrambled tuple)")
    fig.suptitle("405B arity x rank diagonal-enrichment audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(best_df: pd.DataFrame, tests_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 405B Arity Diagonal-Dominance Summary",
        "",
        "This audit recomputes off-diagonal tuple sets from the saved 405B arity cache.",
        "The claim tested here is diagonal enrichment/dominance, not exclusive activation.",
        "Higher-arity relations can contain lower-rank shadows.",
        "",
        "D(r,k) = random-minus-true-relation entropy gap - random-minus-scrambled entropy gap.",
        "",
        "## Best-Layer Cells",
    ]
    for mode in MODES:
        lines.append(f"\n### {MODES[mode]}")
        sub = best_df[best_df["mode"] == mode].copy()
        for arity in ARITIES:
            expected = arity if mode == "argument_subsets" else arity + 1
            diag = sub[(sub["relation_arity"] == arity) & (sub["k"] == expected)]
            off = sub[(sub["relation_arity"] == arity) & (sub["k"] != expected)]
            if diag.empty:
                lines.append(f"- r={arity}: expected k={expected} is undefined.")
                continue
            diag_row = diag.sort_values("mean_D_gt_minus_scrambled_gap", ascending=False).iloc[0]
            max_off = off["mean_D_gt_minus_scrambled_gap"].max() if not off.empty else np.nan
            margin = float(diag_row["mean_D_gt_minus_scrambled_gap"] - max_off) if np.isfinite(max_off) else np.nan
            lines.append(
                f"- r={arity}, expected k={expected}: D={diag_row['mean_D_gt_minus_scrambled_gap']:+.4f} "
                f"at layer {int(diag_row['layer'])}; max off-diagonal={max_off:+.4f}; margin={margin:+.4f}."
            )
    lines.append("\n## Dominance Tests")
    if tests_df.empty:
        lines.append("No dominance tests were available.")
    else:
        key = tests_df[tests_df["comparison_group"] == "max_off_diagonal"].copy()
        for _, row in key.iterrows():
            lines.append(
                f"- {row['mode']}, r={int(row['relation_arity'])}, expected k={int(row['expected_k'])}: "
                f"margin={row.get('diagonal_margin', np.nan):+.4f}, "
                f"p={row.get('margin_p_greater', np.nan):.3g}, "
                f"q={row.get('margin_q_value', np.nan):.3g}, "
                f"supported={row.get('dominance_supported')}."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=str(ROOT / "results" / "cache_405b_relational_arity_benchmark"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    p.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    p.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    p.add_argument("--projection-seed", type=int, default=42)
    p.add_argument("--min-gap", type=int, default=2)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--live-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    layers = parse_int_list(args.layers, DEFAULT_LAYERS)
    ranks = parse_int_list(args.ranks, DEFAULT_RANKS)
    prompt_ids = prompt_ids_from_cache(cache_dir)
    jobs = [
        (str(cache_dir), q, layers, ranks, int(args.proj_dim), int(args.projection_seed), int(args.min_gap))
        for q in prompt_ids
    ]
    prompt_rows: List[Dict[str, object]] = []
    with Pool(processes=int(args.workers)) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_prompt, jobs, chunksize=4), start=1):
            prompt_rows.extend(rows)
            if i % int(args.live_every) == 0 or i == len(jobs):
                print(f"[diagonal] processed {i}/{len(jobs)} rows={len(prompt_rows)}", flush=True)

    prompt_df = pd.DataFrame(prompt_rows)
    cell_df = summarize_cells(prompt_df)
    best_df = best_cell_table(cell_df)
    tests_df = dominance_tests(prompt_df, best_df)
    all_out = pd.concat([cell_df, best_df.assign(row_type="best_cell"), tests_df], ignore_index=True, sort=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "405b_arity_diagonal_dominance.csv"
    prompt_path = results_dir / "405b_arity_diagonal_dominance_prompt_rows.csv"
    fig_path = figures_dir / "fig_405b_arity_x_rank_diagonal_heatmap.png"
    summary_path = results_dir / "405b_arity_diagonal_dominance_summary.md"
    prompt_df.to_csv(prompt_path, index=False)
    all_out.to_csv(csv_path, index=False)
    plot_heatmap(best_df, fig_path)
    write_summary(best_df, tests_df, summary_path)
    print(f"[diagonal] wrote {csv_path}", flush=True)
    print(f"[diagonal] wrote {fig_path}", flush=True)
    print(f"[diagonal] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
