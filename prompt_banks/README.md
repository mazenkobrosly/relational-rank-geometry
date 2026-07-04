# Prompt Banks

This directory contains the raw JSONL prompt banks for the public replication package.

- `controlled_arity_prompts.jsonl`: controlled arity prompts from the main arity assay.
- `nonce_predicate_prompts.jsonl`: arity-neutral nonce-predicate control prompts.
- `query_swap_prompts.jsonl`: query-slot swap control prompts.
- `multi_template_8b_prompts.jsonl`: multi-template reviewer-rescue control prompts.

The multi-template bank contains 400 prompts: 100 each for arities 3, 4, 5, and 6, split evenly across the `registry`, `lab_notes`, `ledger`, `compact`, and `audit` templates. The prompts use arity-neutral predicates and omit the explicit `Relation arity:` line.
