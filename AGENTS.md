## Harness

- For any `network/hrh_final` experiment, metric report, or reproduction-related documentation, start with `HANDOFF.md` and use `RUNBOOK.md` as the only executable command source. State the evidence level, verify the required external asset and split identities, and stop a reproduction claim at a missing or mismatched gate.
- Keep TRAIN/VAL and TEST workflows separate. Do not use TEST for selection or tuning, and preserve the frozen-method restrictions in `results/reports/final_method_manifest.json`.
