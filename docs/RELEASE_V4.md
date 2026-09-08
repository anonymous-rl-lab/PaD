# GitHub reproducibility release v4

Paper v08, raw/prepared data, formal reference records, the mechanism protocol
(including its historical version field and mask salt), and the seven-weight /
five-regularizer selection rule are unchanged.

Changes from v3:

- Nonfinite values in the local readout primitive are replaced with safe zeros
  after recording their availability masks, before arithmetic. Finite inputs
  retain the original computation.
- The default `verify` command includes twelve readout/missingness regressions.
- The feature schema points to `experiments/jonikas/external.py:data()` and
  `data/prepared/frozen_P.json`.
- Current documentation and release metadata identify v4; historical protocol
  identifiers are deliberately retained.

The companion toolkit 0.1.1 also validates reverse-pair grouping for custom CV
and omits author metadata from source and built distributions. See the toolkit's
release notes for its public API changes.

`docs/SOURCE_MAP.json` retains the original archival provenance. Its unchanged
core AST records remain valid. Two previously copied files are now explicitly
listed as adaptations, preserving their original source hash, current hash and
change reason. `MANIFEST_SHA256.json` covers this release.
