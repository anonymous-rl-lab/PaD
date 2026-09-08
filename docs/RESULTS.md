# Paper result map

The paper is frozen at v08; `PAPER_FREEZE.json` identifies the exact archive. `verify` recomputes the quantities below from delivered records. PaD, P, and internal controls retain their original column names in source evidence to preserve provenance.

| Paper evidence | Source in this repository | Rebuild |
|---|---|---|
| 120 mechanism learners; B1 certification; B2 equivalence | `reference/mechanism/formal/*/result.json`, `test_predictions.csv.gz`; aggregate CSVs | `mechanism` |
| All raw decoder checks, including failures and tails | `reference/mechanism/formal/*/test_decoder_diagnostics.csv.gz` | `mechanism` |
| Noise scale selection and independent confirmation | `reference/mechanism/calibration/` | `calibrate` |
| Costanzo main external comparison | `reference/costanzo/pair_predictions.csv`, `method_summary.csv`, `selected/*.json` | `yeast --dataset costanzo`; `external --dataset costanzo` |
| Costanzo internal nonlinear controls | `reference/costanzo/internal_controls.csv` | `yeast --dataset costanzo` |
| Jonikas ordered-reference ranking | `reference/jonikas/pair_predictions.csv`, `method_summary.csv`, `selected/*.json` | `yeast --dataset jonikas`; `external --dataset jonikas` |
| Réd original scores and seeds | `reference/costanzo/red/`, `reference/jonikas/red/` | `red --dataset DATASET` |
| Known directions and ranking changes | `reference/jonikas/known_direction_records.csv`, `ranking_changes.csv` | Derived from refitted ordered predictions |
| Pair-level context availability | `reference/background/` | `background` |

## Quantities that must remain separate

- B1's empirical gap to a fitted additive control is not its certified gap to all additive classifiers. The most adverse fitted risk bound is **0.0953371444**, below the applicable additive lower bound **0.1409257154**; the certified minimum margin is **0.0455885710**.
- All nine B1 classification certificates pass; auxiliary ranking certificates do not. The formal family remains M=30 even for a partial local reproduction.
- Costanzo: P **597/657**, PaD **611/657**, matched nonlinear additive **613/657**, pure RBF additive **606/657**. This panel does not establish that nonadditive scoring is necessary.
- Jonikas: PaD **0.881114**, matched additive **0.827664**, pure RBF additive **0.810982** AUROC on all 168 ordered reference records.
- Background availability: PaD AUROC **0.881114 / 0.712018 / 0.871720** at **100% / 50% / 0%**. Matched additive: **0.827664 / 0.732426 / 0.807580**. The 50% decline and surviving 0% joint advantage restrict biological mechanism attribution.

No reference records were deleted to improve any metric. Removing historical search caches does not remove matched controls or negative results.
