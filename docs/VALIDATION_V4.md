# Release v4 validation

`RELEASE_VALIDATION.json` records the maintenance validation for this release.
It replaces the earlier packaging validation summary; original scientific data,
reference results and protocol files stay byte-identical.

The source and installed toolkit suites pass 25 tests. The GitHub verify command
includes 12 missingness/exchange regression cases and rescoring of every retained
formal prediction. A fresh four-learner B1 block reproduces all predictions exactly.
The public API reruns the full 5/84 outer-fold PaD biological evaluations and
preserves 611/657 and AUROC 0.8811143505021056. A fresh smoke run executes four
mechanism fits, six external-model fits and the raw Costanzo aggregation.

The published archive carries updated file hashes; adapted readout and feature
schema files retain original and current hashes with reasons in SOURCE_MAP.json.
The seven originally checked core ASTs remain unchanged.

Validation is on Linux / Python 3.12. Full historical external training, all
mechanism blocks, every supported Python version and public hosting-account
anonymity are outside this maintenance validation scope.
