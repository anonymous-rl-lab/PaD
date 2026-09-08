# Inputs and attribution

## Prepared PaD inputs

`data/prepared/Costanzo.{csv,npz}` (657 records) and `KEGG.{csv,npz}` (168 records) are the exact frozen feature interfaces used in the reported fits. NPZ keys are `p`, `xp`, `xpr`, `odd`, `xd`, `xdr`, `y`, and `directional`; CSVs retain gene pairs, fold/group identities, references and measured metadata. `frozen_P.json` records the original five-feature coefficients and scale.

P summarizes third-party single-perturbation expression responses. The expression study is [GSE42527](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42527), with normalized log2-ratio measurements. The complete original expression download, source-specific adapter history and inherited Proposer training ancestry are not included. The prepared inputs and original attribution are retained; no claim of rebuilding that ancestry is made.

## Costanzo 2016

Source: [Costanzo et al. 2016](https://pubmed.ncbi.nlm.nih.gov/27708008/), [TheCellMap dataset](https://thecellmap.org/yeast/costanzo2016/).

- `data/costanzo/costanzo_matched_records.json`: source assay records for the 657 matched candidate pairs.
- `red_raw_within_panel.csv.gz`: all selected DMA30 assay rows within the endpoint panel, retaining single and double fitness measurements.
- `red_input.npz`: aggregated matrices used by Réd; rebuilt by `prepare-costanzo`.
- `frozen_proposer.json`: inherited per-pair P outputs and metadata.
- `INHERITED_SOURCES.json`: original source URL, archive hash, extraction fields, label semantics and retained provenance.

The official full raw archive is not bundled. To repeat extraction from it, download the URL recorded in `INHERITED_SOURCES.json` and run:

```bash
python reproduce.py prepare-costanzo --archive /path/to/official_archive.zip
```

Its SHA256 must equal `05adf2aa309e5336cd1c3045eb033a2bc7b24e6f249d57a0cc25549ec6772522`. The default command instead aggregates the bundled assay subset. Neither route retrains Proposer.

## Jonikas 2009

Source: [Jonikas et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2877488/). The original MAT and reference lists were distributed through the [Réd author repository](https://github.com/biolab/red), commit `dec800b3f15fb557ee585a3621cd44b7118e5cc9`.

`data/jonikas/080930a_DM_data.mat` retains the single/double UPR phenotype arrays, null expectation, measured-count information and gene names. `KEGG_ordered.txt` and `KEGG_nonordered.txt` define the 21/147 main reference records. The small glycan reference text files remain because the original Réd reproduction evaluates them in the same pass; no separate exploratory glycan feature/cache pipeline is included. File hashes are recorded in `provenance.json`.

The [Battle et al. pathway reconstruction work](https://doi.org/10.1038/msb.2010.27) is prior art for the biological task; its MATLAB code is not a fitted competitor in these experiments and is not included as unused code in this release. Published comparison values are not substituted for newly fitted predictions.

## Mechanism data

Generated deterministically by `experiments/mechanism/engine.py` using `configs/mechanism.json`. Every formal record is generated through raw expression and phenotype measurements, then the complete encoders. Hidden variables are retained in diagnostic reference tables for verification but are not model inputs. Selection and confirmation use independent random streams.

## Rights and provenance

Original data retain their providers' terms. Source citations and retained provenance do not assign new redistribution rights. `docs/SOURCE_MAP.json` maps preserved files and unchanged extracted symbols to the archival source; the current release manifest authenticates the deliverable itself. `third_party/red/` carries the original copyright and GPL license.
