# Reviewer-Rescue Prompt Controls

These JSONL prompt banks address template-artifact concerns in the Plucker sign entropy paper.

Controls:
- `nonce_predicate_control`: removes arity-coded predicate names and the explicit arity line.
- `generic_predicate_control`: same, using generic `RELxx` names.
- `query_swap_nonce_control`: changes which relation is queried while also using nonce predicates.
- `template_generalization_control`: mixes five surface templates and labels three as dev, two as heldout.

All generated controls use r=3..6, 100 prompts per arity, and 20 relation instances per prompt.
Forward passes are not run by this builder. Use `scripts/run_reviewer_rescue_8b_controls.sh` on a GPU pod.

Manifest:

```json
{
  "arities": [
    3,
    4,
    5,
    6
  ],
  "builder": "analysis/build_reviewer_rescue_controls.py",
  "forward_pass_required": true,
  "n_per_arity": 100,
  "n_relation_instances": 20,
  "purpose": "controls for arity-coded predicate names, query-slot salience, and template artifacts",
  "seed": 40577,
  "total_rows_per_bank": {
    "arity_generic_predicate_r3_r6.jsonl": 400,
    "arity_nonce_predicate_r3_r6.jsonl": 400,
    "arity_query_swap_nonce_r3_r6.jsonl": 400,
    "arity_template_generalization_r3_r6.jsonl": 400
  }
}
```
