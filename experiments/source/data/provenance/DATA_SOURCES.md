# Prepared-input provenance and scope

The files actually supplied here are `data/costanzo/Costanzo.csv`, `Costanzo.npz` (657 records), and `data/jonikas/KEGG.csv`, `KEGG.npz` (168 records). They are the frozen prepared P/D interfaces used by the reported learning procedures. They are not the full raw biological archives.

P originates from the Kemmeren expression perturbation study (GSE42527). Costanzo supplies growth-fitness perturbation measurements; Jonikas supplies unfolded-protein-response measurements and the retained KEGG reference panel, following the original study and Réd/Battle pathway-analysis sources cited in the paper. The accompanying provenance JSON files retain upstream paths and hashes as historical attribution; such paths do not mean those raw files or the broader repository are included here.

The root paper archive retains the frozen Proposer feature coefficients in `verification/frozen_P.json`. The historical Proposer training ancestry and full raw-expression reconstruction are not supplied. No mechanism-refitting engine, installable distribution or third-party full-study code is implied by this minimal prepared-input archive. Original datasets retain their providers' terms; no new redistribution license is asserted.
