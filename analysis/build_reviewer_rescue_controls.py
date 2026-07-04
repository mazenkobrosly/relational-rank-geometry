#!/usr/bin/env python3
"""Build reviewer-rescue arity prompt controls.

These prompt banks are designed to address the most direct workshop-reviewer
objection to the controlled arity result: the original symbolic benchmark used
predicate names and prompt text that made arity too easy to infer from surface
form (e.g. TRIADMAP, PENTASYS, and an explicit "Relation arity" line).

The generated controls preserve the same relational tuple structure while
removing arity-coded predicate names and adding query/template perturbations.
They intentionally only build prompt banks; forward passes still need to be run
with the usual relational-arity benchmark scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_8b_relational_arity_benchmark import (  # noqa: E402
    build_task_bank,
    relation_expr,
    rotated_distractor_choices,
    write_jsonl,
)


ARITIES = [3, 4, 5, 6]
LETTERS = ["A", "B", "C", "D"]

NONCE_SYLLABLES = [
    "vorn", "mip", "taz", "lor", "nex", "pav", "sul", "dren",
    "keth", "zoma", "quor", "bel", "rith", "fex", "naru", "galt",
    "seph", "luma", "torv", "brin", "azek", "mora", "cav", "dax",
]


def stable_int(*parts: object) -> int:
    raw = "::".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def nonce_predicate(prompt_id: str, variant: str) -> str:
    """Return an arity-neutral nonce predicate with no numeric or Greek hint."""
    h = stable_int(prompt_id, variant)
    a = NONCE_SYLLABLES[h % len(NONCE_SYLLABLES)]
    b = NONCE_SYLLABLES[(h // 97) % len(NONCE_SYLLABLES)]
    c = NONCE_SYLLABLES[(h // 7919) % len(NONCE_SYLLABLES)]
    return (a + b + c).upper()


def generic_predicate(prompt_id: str, variant: str) -> str:
    h = stable_int(prompt_id, variant)
    return f"REL{string.ascii_uppercase[h % 26]}{string.ascii_uppercase[(h // 31) % 26]}"


def update_predicate(row: MutableMapping, predicate: str) -> None:
    for rel in row["relations"]:
        rel["predicate"] = predicate
        rel["relation_expr"] = relation_expr(predicate, rel["arguments"])
    target_idx = int(row["target_relation_index"])
    row["target_relation"] = row["relations"][target_idx]


def choices_for_target(row: Mapping, answer_letter: str) -> dict:
    target = row["relations"][int(row["target_relation_index"])]
    choices_raw = rotated_distractor_choices(target["arguments"], row["distractor_entities"])
    distractor_iter = iter(choices_raw[1:])
    choices = {}
    for letter in LETTERS:
        choices[letter] = choices_raw[0] if letter == answer_letter else next(distractor_iter)
    return choices


def render_prompt(row: Mapping, template_id: str) -> str:
    facts = "\n".join(f"  {rel['fact_label']}: {rel['relation_expr']}" for rel in row["relations"])
    target = row["target_relation"]
    target_line = f"{target['fact_label']} = {target['relation_expr']}"
    if template_id == "registry":
        return (
            "A registry contains ordered relation calls. Each record has one predicate "
            "and one ordered argument tuple.\n\n"
            f"Records:\n{facts}\n\n"
            f"Requested record: {target_line}\n"
            "Which option gives the requested record's arguments in exactly the same order?"
        )
    if template_id == "lab_notes":
        return (
            "Read the following lab notes. Each line names a record and then gives an "
            "ordered call.\n\n"
            f"{facts}\n\n"
            f"The record to recover is {target_line}.\n"
            "Choose the option whose tuple matches that record exactly."
        )
    if template_id == "ledger":
        return (
            "The ledger below stores ordered tuples inside relation calls. Preserve "
            "the order; do not sort the symbols.\n\n"
            f"Ledger entries:\n{facts}\n\n"
            f"Entry under query: {target_line}\n"
            "Which answer option repeats the entry's argument tuple?"
        )
    if template_id == "compact":
        return (
            "For each keyed item below, the expression after the colon is an ordered "
            "relation call.\n\n"
            f"{facts}\n\n"
            f"Key to inspect: {target_line}\n"
            "Select the exact ordered argument tuple for that key."
        )
    if template_id == "audit":
        return (
            "Audit this table of relation-call facts. The tuple order is meaningful.\n\n"
            f"{facts}\n\n"
            f"Audit target: {target_line}\n"
            "Which option lists the target arguments without changing their order?"
        )
    raise ValueError(f"unknown template_id: {template_id}")


def normalize_row(row: MutableMapping, control_name: str, template_id: str) -> None:
    row["prompt"] = render_prompt(row, template_id)
    row["choices"] = choices_for_target(row, str(row["answer"]))
    row["task_family"] = "reviewer_rescue_relational_arity"
    row["style_condition"] = control_name
    row.setdefault("control_metadata", {})
    row["control_metadata"].update(
        {
            "control_name": control_name,
            "template_id": template_id,
            "no_explicit_arity_line": True,
            "arity_neutral_predicate": True,
            "relation_instances_per_prompt": len(row["relations"]),
        }
    )
    row["metadata_schema"] = "reviewer-rescue controls; spans are added after tokenization"


def filter_arities(rows: Sequence[Mapping]) -> List[dict]:
    return [deepcopy(r) for r in rows if int(r.get("relation_arity", 0)) in ARITIES]


def make_nonce(rows: Sequence[Mapping], generic: bool = False) -> List[dict]:
    out = []
    name = "generic_predicate_control" if generic else "nonce_predicate_control"
    for row in filter_arities(rows):
        old_pred = row["relations"][0]["predicate"]
        pred = generic_predicate(row["prompt_id"], name) if generic else nonce_predicate(row["prompt_id"], name)
        update_predicate(row, pred)
        normalize_row(row, name, "registry")
        row["control_metadata"].update({"old_predicate": old_pred, "new_predicate": pred})
        row["prompt_id"] = f"{name}_{row['prompt_id']}"
        out.append(row)
    return out


def make_query_swap(rows: Sequence[Mapping], use_nonce: bool = True) -> List[dict]:
    out = []
    name = "query_swap_nonce_control" if use_nonce else "query_swap_control"
    for row in filter_arities(rows):
        old_idx = int(row["target_relation_index"])
        n = len(row["relations"])
        new_idx = (old_idx + 7) % n
        row["target_relation_index"] = new_idx
        if use_nonce:
            pred = nonce_predicate(row["prompt_id"], name)
            update_predicate(row, pred)
        else:
            row["target_relation"] = row["relations"][new_idx]
        normalize_row(row, name, "registry")
        row["control_metadata"].update(
            {
                "old_target_relation_index": old_idx,
                "new_target_relation_index": new_idx,
                "query_swapped": True,
            }
        )
        row["prompt_id"] = f"{name}_{row['prompt_id']}"
        out.append(row)
    return out


def make_template_generalization(rows: Sequence[Mapping]) -> List[dict]:
    out = []
    templates = ["registry", "lab_notes", "ledger", "compact", "audit"]
    for i, row in enumerate(filter_arities(rows)):
        template_id = templates[i % len(templates)]
        pred = nonce_predicate(row["prompt_id"], f"template_generalization_{template_id}")
        old_pred = row["relations"][0]["predicate"]
        update_predicate(row, pred)
        normalize_row(row, "template_generalization_control", template_id)
        row["control_metadata"].update(
            {
                "old_predicate": old_pred,
                "new_predicate": pred,
                "template_split": "dev" if template_id in {"registry", "lab_notes", "ledger"} else "heldout",
            }
        )
        row["prompt_id"] = f"template_{template_id}_{row['prompt_id']}"
        out.append(row)
    return out


def write_readme(out_dir: Path, manifest: Mapping) -> None:
    lines = [
        "# Reviewer-Rescue Prompt Controls",
        "",
        "These JSONL prompt banks address template-artifact concerns in the Plucker sign entropy paper.",
        "",
        "Controls:",
        "- `nonce_predicate_control`: removes arity-coded predicate names and the explicit arity line.",
        "- `generic_predicate_control`: same, using generic `RELxx` names.",
        "- `query_swap_nonce_control`: changes which relation is queried while also using nonce predicates.",
        "- `template_generalization_control`: mixes five surface templates and labels three as dev, two as heldout.",
        "",
        "All generated controls use r=3..6, 100 prompts per arity, and 20 relation instances per prompt.",
        "Forward passes are not run by this builder. Use `scripts/run_reviewer_rescue_8b_controls.sh` on a GPU pod.",
        "",
        "Manifest:",
        "",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(ROOT / "data" / "reviewer_rescue"))
    p.add_argument("--n-per-arity", type=int, default=100)
    p.add_argument("--n-relation-instances", type=int, default=20)
    p.add_argument("--seed", type=int, default=40577)
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = build_task_bank(args.n_per_arity, args.n_relation_instances, seed=args.seed)
    base = filter_arities(base)

    banks = {
        "arity_nonce_predicate_r3_r6.jsonl": make_nonce(base, generic=False),
        "arity_generic_predicate_r3_r6.jsonl": make_nonce(base, generic=True),
        "arity_query_swap_nonce_r3_r6.jsonl": make_query_swap(base, use_nonce=True),
        "arity_template_generalization_r3_r6.jsonl": make_template_generalization(base),
    }

    manifest = {
        "builder": "analysis/build_reviewer_rescue_controls.py",
        "seed": int(args.seed),
        "arities": ARITIES,
        "n_per_arity": int(args.n_per_arity),
        "n_relation_instances": int(args.n_relation_instances),
        "total_rows_per_bank": {name: len(rows) for name, rows in banks.items()},
        "forward_pass_required": True,
        "purpose": "controls for arity-coded predicate names, query-slot salience, and template artifacts",
    }
    for name, rows in banks.items():
        write_jsonl(out_dir / name, rows)
        print(f"[reviewer-rescue] wrote {out_dir / name} rows={len(rows)}")
    (out_dir / "manifest_reviewer_rescue_controls.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(out_dir, manifest)
    print(f"[reviewer-rescue] wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
