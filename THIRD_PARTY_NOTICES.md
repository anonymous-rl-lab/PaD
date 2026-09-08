# Third-party notices

## Réd

Copyright (C) 2013 Marinka Zitnik. Original repository: https://github.com/biolab/red, commit `dec800b3f15fb557ee585a3621cd44b7118e5cc9`.

`third_party/red/red_py3.py` is the existing Python-3-compatible port used in the reported experiments. Its original copyright and GPL-3.0-or-later notice are retained, with the full license in `third_party/red/LICENSE.txt`. The port changes Python 2 syntax/import compatibility; the released asymmetric downstream scoring behavior remains unchanged. No Réd scoring improvement is introduced by this packaging release.

## Other dependencies and data

NumPy, SciPy, pandas, scikit-learn, XGBoost, CatBoost, joblib and threadpoolctl are installed from their distributions; their code is not vendored in this ZIP and retains the applicable licenses. Biological inputs and reference lists retain the source terms described in `docs/DATA_SOURCES.md`.

## Inherited Proposer

The five features, coefficient vector and historical predictions predate the current packaging work. They are frozen evidence interfaces, not newly trained components of release v4. Their exact source mapping is retained in `docs/SOURCE_MAP.json` and the data provenance JSON files.
