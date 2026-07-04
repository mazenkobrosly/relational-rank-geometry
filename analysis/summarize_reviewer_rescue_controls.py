#!/usr/bin/env python3
"""Summarize reviewer-rescue arity control runs.

This scans a run root for per-control diagonal-dominance CSVs produced by the
model-specific `run_*_arity_diagonal_dominance.py` scripts and produces compact
tables/figures suitable for the paper packet.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_model_and_control(path: Path) -> Dict[str, str]:
    parent = path.parent.name
    # Expected: results_8B_arity_nonce_predicate_r3_r6
    m = re.match(r"results_(.+?)_(arity_.+)", parent)
    if m:
        return {"model": m.group(1), "control": m.group(2)}
    fname = path.name
    model = fname.split("_", 1)[0].upper()
    return {"model": model, "control": parent}


def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = infer_model_and_control(path)
    best = df[
        (df.get("row_type", "") == "best_cell")
        & (df.get("mode", "") == "argument_subsets")
        & (pd.to_numeric(df.get("relation_arity"), errors="coerce") == pd.to_numeric(df.get("k"), errors="coerce"))
    ].copy()
    if best.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    dom = df[
        (df.get("row_type", "") == "row_dominance")
        & (df.get("mode", "") == "argument_subsets")
        & (df.get("comparison_group", "") == "max_off_diagonal")
    ].copy()
    for _, row in best.iterrows():
        arity = int(row["relation_arity"])
        drow = dom[pd.to_numeric(dom["relation_arity"], errors="coerce") == arity]
        drow = drow.iloc[0] if not drow.empty else None
        rows.append(
            {
                **meta,
                "relation_arity": arity,
                "k": int(row["k"]),
                "selected_layer": int(row["layer"]),
                "D_rr": float(row["mean_D_gt_minus_scrambled_gap"]),
                "D_ci_low": float(row.get("D_ci_low", np.nan)),
                "D_ci_high": float(row.get("D_ci_high", np.nan)),
                "positive_fraction": float(row.get("D_positive_fraction", np.nan)),
                "q_value": float(row.get("D_q_value", np.nan)),
                "diagonal_margin": float(drow.get("diagonal_margin", np.nan)) if drow is not None else np.nan,
                "margin_q_value": float(drow.get("margin_q_value", np.nan)) if drow is not None else np.nan,
                "dominance_supported": bool(drow.get("dominance_supported", False)) if drow is not None else False,
                "source_csv": str(path),
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    models = list(dict.fromkeys(df["model"].astype(str)))
    controls = list(dict.fromkeys(df["control"].astype(str)))
    nrows = len(models)
    ncols = len(controls)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False, constrained_layout=True)
    vmax = max(0.8, float(np.nanmax(df["D_rr"].to_numpy(dtype=float))) if not df.empty else 0.8)
    for i, model in enumerate(models):
        for j, control in enumerate(controls):
            ax = axes[i][j]
            sub = df[(df["model"] == model) & (df["control"] == control)].copy()
            mat = np.full((4, 1), np.nan)
            labels = [3, 4, 5, 6]
            for ridx, r in enumerate(labels):
                g = sub[sub["relation_arity"] == r]
                if not g.empty:
                    mat[ridx, 0] = float(g["D_rr"].iloc[0])
            im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
            ax.set_title(f"{model}\n{control}", fontsize=9)
            ax.set_xticks([0], ["k=r"])
            ax.set_yticks(range(4), [f"r={r}" for r in labels])
            for ridx, r in enumerate(labels):
                val = mat[ridx, 0]
                if np.isfinite(val):
                    ax.text(0, ridx, f"{val:.3f}", ha="center", va="center", color="white" if val > vmax * 0.45 else "black", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="D(r,r)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Reviewer-Rescue Control Summary", ""]
    if df.empty:
        lines.append("No completed reviewer-rescue control result CSVs were found yet.")
    else:
        lines.append("Completed controls found:")
        lines.append("")
        show = df[["model", "control", "relation_arity", "k", "selected_layer", "D_rr", "D_ci_low", "D_ci_high", "q_value", "diagonal_margin", "margin_q_value"]].copy()
        for c in ["D_rr", "D_ci_low", "D_ci_high", "q_value", "diagonal_margin", "margin_q_value"]:
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        lines.append(show.to_markdown(index=False))
        lines.append("")
        lines.append("Paper sentence if the nonce/query-swap rows remain positive with q<0.05:")
        lines.append("")
        lines.append("> The arity diagonal persists under arity-neutral predicate labels and query-swapped prompts, reducing the risk that the statistic is driven by predicate-name arity leakage or a fixed queried template slot.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames = []
    for path in sorted(run_root.glob("results_*/*_arity_diagonal_dominance.csv")):
        one = load_one(path)
        if not one.empty:
            frames.append(one)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reviewer_rescue_control_summary.csv"
    df.to_csv(csv_path, index=False)
    if not df.empty:
        plot(df, out_dir / "fig_reviewer_rescue_controls.png")
    write_summary(df, out_dir / "reviewer_rescue_control_summary.md")
    print(f"[reviewer-rescue-summary] wrote {csv_path} rows={len(df)}")


if __name__ == "__main__":
    main()
