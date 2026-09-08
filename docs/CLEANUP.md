# Release cleanup decisions

The source experiment archive remains untouched. This release has one shared implementation and separate input, reference and generated-output directories. No scientific result, sample, prescribed seed or comparator was removed.

| Removed from the release | Files | Original bytes | Why |
|---|---:|---:|---|
| Repeated archive inventories, historical candidates and redundant supporting files | 149 | 15,544,616 | Replaced by a current manifest, compact provenance, explicit commands and formal evidence. |
| Duplicate modules and obsolete/debug/history entry points | 26 | 169,900 | Shared kernels/selector/encoder are extracted once; old architecture searches and pilots are not required to reproduce the frozen protocol. |
| Debug logs and repeated fit event/append logs | 821 | 70,784,810 | Formal compute summaries and selected-config records are retained; new runs write a single ledger. |
| Generated external model pickles | 455 | 108,444,764 | Recreated by complete external training; their formal predictions and selected configurations are retained. |
| Generated encodings, model factors and selection caches; obsolete prepared subpanels | 1194 | 119,699,641 | Rebuilt from raw mechanism streams or prepared biological inputs; independent source measurements are retained. |

## What remains

- Exact prepared feature NPZ/CSV inputs, inherited coefficient/provenance JSON, raw phenotype MAT and assay rows.
- All 30 formal mechanism blocks, 120 learner outcomes and all per-instance test predictions; calibration selection/confirmation records and decoder tails.
- All 35 Costanzo and 420 Jonikas selected-model records, all prescribed Réd scores, and compact complete method/paired-result tables.
- Costanzo internal matched-additive result (613/657), failed auxiliary ranking certificates, and nonmonotone context-availability results.
- Source-file hashes, unchanged core-function AST hashes, current release manifest, and the unchanged paper archive identifier.

CSV files marked `.csv.gz` are compressed losslessly; their uncompressed SHA256 values are preserved in `SOURCE_MAP.json`. The Réd raw input NPZ and biological prepared arrays are not rounded or converted.

The historical timing totals retain their original scope, which may include earlier smoke/pilot work. New generated ledgers are distinct. Removing repeated timing logs does not redefine those reported totals.

The paper PDFs and TeX are frozen in the separately delivered paper v08 archive; this source repository records its hash instead of copying the manuscript and earlier paper versions.
