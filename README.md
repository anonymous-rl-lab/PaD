# PaD — when perturbation context changes directional evidence

**Anonymous GitHub reproducibility release v4. Paper frozen at v08.**

[Revision notes](docs/RELEASE_V4.md) · [Anonymous distribution](docs/ANONYMITY.md)

PaD combines a frozen Proposer channel with double-perturbation evidence. This repository contains the shared learner, three experiment suites, required prepared inputs, raw phenotype inputs, and complete formal prediction records. It replaces the bulky experiment delivery with a runnable source release; paper results and the learning rules are unchanged.

[中文说明](README_CN.md) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Data and provenance](docs/DATA_SOURCES.md) · [Result map](docs/RESULTS.md)

## Quick start

Tested with Python 3.12 on Linux, CPU only.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python reproduce.py verify
```

`verify` recomputes the metrics of **all 120 mechanism learners**, the nine B1 classification certificates, paired B2 equivalence results, both yeast comparisons, and the context-availability diagnostic. It also regenerates deterministic encoder checks and checks file integrity. It does **not** retrain the models.

To run model fits, install the remaining external-model dependencies:

```bash
python -m pip install -r requirements.txt
python reproduce.py smoke
```

All generated files go to `runs/`, which is excluded from Git. Use `python reproduce.py --output /path/to/run-directory COMMAND` to choose another location. The bundled `data/` and `reference/` directories remain read-only inputs.

## Reproduce the experiments

| Experiment | Command | Reproduction boundary |
|---|---|---|
| 120 mechanism training and selection flows | `python reproduce.py mechanism` | Raw generated measurements → complete encoders → nested selection → independent test |
| One complete four-learner block | `python reproduce.py mechanism --scenario B1 --seed 17 --level ideal` | 512 training instances; 4,096 independent test instances |
| Noise selection and independent confirmation | `python reproduce.py calibrate` | Rebuilds the original calibration; the formal schedule uses the delivered, frozen noise levels |
| PaD and both nonlinear additive controls | `python reproduce.py yeast` | Full endpoint-excluded nested refits from the delivered prepared inputs |
| Costanzo external comparison | `python reproduce.py external --dataset costanzo` | Original SVM/XGBoost/CatBoost grids and prescribed seeds |
| Jonikas external comparison | `python reproduce.py external --dataset jonikas --workers 4` | Original grids; 84 held-out unordered pairs |
| Réd comparisons | `python reproduce.py red --dataset costanzo` and `python reproduce.py red --dataset jonikas` | Fresh factorization using the released Réd implementation |
| Jonikas context diagnostic | `python reproduce.py background` | Fresh nested refits at 100%, 50%, and 0% pair-level availability |
| Raw Costanzo panel aggregation | `python reproduce.py prepare-costanzo` | Rebuilds the Réd matrices from the bundled assay rows |

The full external comparisons are substantially more expensive than verification. They are optional local runs, not part of the default GitHub workflow. Model pickles and nested caches are generated on demand and are not shipped.

## Frozen results

| Evidence | Result | Meaning |
|---|---|---|
| B1 classification, ideal and two nonzero noise levels | **9/9 certificates**; minimum certified margin **4.558857 percentage points** | Fitted classifiers exceed the applicable additive-function-class risk bound |
| Costanzo 2016 | **611/657** for PaD | Above all tested external supervised configurations; internal matched additive control: **613/657** |
| Jonikas 2009 | **AUROC 0.881114** for PaD | Above the tested external supervised configurations on 168 reference records |

These are different tasks. The mechanism classification certificate is not a certificate for the Jonikas reference-ranking task. The matched Costanzo control, failed auxiliary ranking certificates, and the negative mixed-context result are retained in full. See [the result map](docs/RESULTS.md).

## Repository layout

- `src/pad/`: one shared copy of the frozen kernels, selector, Proposer features, and readout functions.
- `experiments/`: mechanism generation/calibration, yeast adapters, and external-model training.
- `configs/`: original finite grids, physical settings, seeds, masks, and certification family.
- `data/`: prepared PaD inputs and source phenotype/reference data.
- `reference/`: formal predictions, selected configurations, calibration records, and result summaries.
- `verification/`: evidence rescore and encoder/solver checks.
- `third_party/red/`: attributed Python 3 port and original GPL license.
- `docs/`: methods-to-files mapping, provenance, cleanup decisions, and paper freeze record.

The release starts real-data PaD fitting from the preserved P/D feature interface. It does not reconstruct the complete historical training ancestry of the inherited Proposer or regenerate all expression-derived features from a raw GEO download. The mechanism suite does execute the full raw-measurement-to-encoder path. These boundaries are documented separately from result verification.

## Attribution and use

Please cite the accompanying frozen manuscript and the original datasets/methods listed in [DATA_SOURCES.md](docs/DATA_SOURCES.md). Réd remains GPL-3.0-or-later. This packaging release does not assign a new license to PaD code or to third-party biological data; see [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

After fitting, run `python reproduce.py report` (with the same `--output` root) to build current-run method tables and a completion report. It does not fill missing runs with bundled reference predictions.
