# PaD paper-support experiments — v31 integration

The two source-support experiments are integrated here with corrected interpretation and a complete fixed-prediction statistics generator. The learning rule and prepared inputs are unchanged.

## Lightweight checks (no training)

```bash
python code/verify_stored_results.py
python code/recompute_source_statistics.py
python code/recompute_J_intervals.py --pad reference/PaD_jonikas_predictions.csv --joint results/joint_control/jonikas_J_predictions.csv --out rerun_results/paired_J
```

The statistics generator reproduces both 10,000-draw designs from frozen predictions and seeds, including every primary, full-context M and selectivity summary and all leave-one-gene evaluations. `--save-draws` additionally writes the reproducible draw-level outputs. These are descriptive sensitivities of fixed predictions, not a general 95% network-dependence guarantee or independent validation.

## Refitting (separate from verification)

```bash
python code/joint_control_reproduce.py
python code/source_ablation_reproduce.py
python code/source_task_hardening.py
```

Refits write to `rerun_results/`, never overwrite frozen results, and may be redirected with `PAD_OUT`. Source ablations retain complete P and the original 23-coordinate D denominator. `PAD_VARIANTS=SH,SMH,SMHV,FULL` selects named configurations without changing the learner. The full hardening runner refits five configurations and 100 block mismatches per source, then invokes the same statistics generator. Its parallel refitting path requires Linux/WSL (POSIX fork); light statistical replay is platform-independent. Use one BLAS thread per worker.

## Contents

- `data/costanzo/` and `data/jonikas/`: frozen prepared inputs, not the full raw-study archives.
- `protocols/`: unchanged historical protocols, plus current scope clarification. A historically named shuffle p field is interpreted only as a plus-one mismatch exceedance fraction, not conditional inference.
- `results/paper_source_predictions.csv`: all reported source configurations, including SMHV.
- `results/hardening_static_predictions.csv`: the five hardening configurations.
- `results/hardening_shuffle_plans.npz` and `hardening_shuffle_results.csv`: exact pair-orbit plans and 200 retained refit rows.
- `results/hardening_*summary.csv`: regenerated numerical summaries with an executable source.
- `results/paired_J_intervals.csv`: PaD-minus-J paired descriptive intervals.
- `reference/`: retained reference predictions and input/score alignment checks.

The only primary M comparison is SH→SMH. M's full-context comparison is supplementary and has no same-context mismatch experiment. Selectivity ranges spanning zero, V's DIE2 sensitivity, the saturated direction endpoint and the alternative SMHV loss are retained. No fixed M/V task allocation or confirmed double dissociation is asserted. All results use the same development panel.
