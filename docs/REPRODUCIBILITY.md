# Reproduction contract

## Inputs, fitting, and references are separate

`data/` contains read-only inputs. `reference/` contains the preserved formal evidence. Every command writes to `runs/` or the root selected by `--output`. Full training commands calculate predictions from input measurements/features; they do not fit against reference predictions.

The formal mechanism runner uses the **published frozen calibration values**, not the output of a newly rerun calibration. `calibrate` independently reproduces how those levels were obtained. This keeps selected levels fixed across releases; it is not a new noise-level search.

The random stream keys retain phase, purpose, scenario, seed and noise-level identity. Formal runs always use the original `formal` stream. The `smoke` command uses a separate `release-smoke` stream. Four learners share the same test instances; reverse predictions do not count as additional independent instances.

## Levels of reproduction

1. **Evidence verification:** `verify` losslessly authenticates retained inputs and records; rescales no data; recalculates all 120 learner metrics and 455 external selected-model prediction records; recomputes the nine B1 certificates and B2 intervals; regenerates 240 formal raw encoder samples and deterministic theory cases. This command does not load saved model weights or perform full training.
2. **Executable smoke:** `smoke` fits all four mechanism learners on a separate small dataset, compares the spectral solver against an independent direct solve, executes all three external learner families on each yeast dataset, and reconstructs Costanzo Réd matrices from raw panel rows.
3. **Complete model reproduction:** `mechanism`, `yeast`, `external`, and `red` rebuild the corresponding models. No delivered nested-score caches or model pickles are needed. `background` rebuilds all six diagnostic fits from prepared inputs, including the 100% arm that previously reused inner-score caches.
4. **Raw biological-data boundary:** the original Jonikas MAT and reference lists, Costanzo assay rows and matched-record JSON, and frozen P/D arrays are included. The complete GEO-expression preprocessing/training ancestry is not reconstructed by this release. Raw assay aggregation does not reconstruct the inherited expression channel.

## Compute and resume

The mechanism schedule is 72 ideal plus 48 noisy training/selection flows: 30 blocks, four learners per block, 512 training and 4,096 test instances. It retains 16,500 candidate-inner-fold evaluations and a reserved certification family of M=30. Full-P RBF is selected separately over five regularizers; the three other learner pools each have 35 candidates.

The external comparison uses 35 selected Costanzo models and 420 selected Jonikas models, plus three Réd factorizations per dataset. Jonikas nested fitting is the longest task. Its `--workers` option controls independent CPU workers; each worker uses one numerical thread. Run it separately from lightweight verification.

Mechanism blocks reuse a completed `result.json` in the same output directory. External training caches are keyed by the protocol, current adapter source and input NPZ. Use a **fresh `--output` directory after modifying code/configuration**, or when a genuinely fresh rerun is desired. Do not combine outputs from different schedules. No training command overwrites the reference evidence.

Examples:

```bash
python reproduce.py --output runs_fresh mechanism
python reproduce.py --output runs_fresh yeast
python reproduce.py --output runs_fresh external --dataset costanzo
python reproduce.py --output runs_fresh external --dataset jonikas --workers 4
python reproduce.py --output runs_fresh red --dataset costanzo
python reproduce.py --output runs_fresh red --dataset jonikas
python reproduce.py --output runs_fresh background
```

The `runs_fresh` directory is an example outside the default ignored `runs/`; exclude any custom output directory from a public commit.

## Selection and measurement semantics

Mechanism normalization is fit within each inner training subset and frozen for test application. Real-data refits retain the historical unlabeled-panel transformations already stored in the prepared arrays, while fitting the linear-component scales within each training subset. The protocols intentionally differ at this measurement boundary; neither test normalization nor kernel width is silently changed in this release.

Costanzo evaluates inherited response-direction labels. Jonikas evaluates 21 positive and 147 other ordered reference records; these are not 147 established reverse-direction labels. The main Jonikas score is the full ordered score, while direction diagnostics compare swapped records.

External methods retain their original grids and selectors. They receive the same prepared information but do not have identical candidate counts or selection rules to PaD. Réd uses its released phenotype formulation and supervision regime. The comparisons support performance against the tested methods, not a universal SOTA claim.

## Outputs

- `runs/mechanism/formal/<block>/`: regenerated raw-encoding arrays, selected model coefficients, candidate scores, full predictions, and per-block result JSON.
- `runs/costanzo/`, `runs/jonikas/`: PaD/control predictions, choices and metrics; external nested records, regenerated caches and models; Réd outputs.
- `runs/mechanism/calibration/`: independent scale estimation, selection, confirmation, tails and error budgets.
- `runs/background/`: masks, selected predictions, metrics and paired changes for all availability levels.
- `runs/verification/`: rescore and smoke reports. These are execution outputs, not additional scientific experiments.

The original full experiment archive remains the archival source. This release deliberately omits historical debug code and generated model/cache bulk while preserving the formal results and their source mapping.

After fitting, run `python reproduce.py report` (with the same `--output` root) to build current-run method tables and a completion report. It does not fill missing runs with bundled reference predictions.
