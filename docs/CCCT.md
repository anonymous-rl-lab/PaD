# Cross-Channel Correspondence Test: methods to files

The CCCT adds one prospectively specified test to the existing Costanzo and
Jonikas panels and to four mechanism control settings. It adds no biological
data and changes no encoder, adapter, kernel family, candidate grid, selector
or split rule. The frozen protocol is `protocols/CCCT_FROZEN_PROTOCOL.md`,
authenticated by `protocols/CCCT_FROZEN_PROTOCOL.sha256`; every output record
carries that hash.

## What the test asks

Is the observed joint-minus-additive gain larger than what the same joint
learner produces on data in which the two channels carry no correspondence
beyond what each carries about the label?

Under `H0 : P independent of D given Y` the Bayes-optimal score is additive
(Proposition CCCT-1), so the population joint-minus-additive gain is zero. The
implication is one way: additivity of the Bayes score does not require
conditional independence. Rejection therefore says the channels are
conditionally dependent in a way the joint kernel converts into ranking gain;
it does not identify the shared background of Section 5.1.

## Files

| Component | File | Protocol section |
|---|---|---|
| Frozen procedure runner, provenance-adaptive split structure | `ccct/frozen.py` | 3, 4.6, 11 |
| Permutation operators (Costanzo, Jonikas, mechanism) | `ccct/operators.py` | 4.1-4.5 |
| Test statistics, permutation summary, Holm | `ccct/stats.py` | 5, 6 |
| Panel driver, placebo-matched reference, equivalence reading | `ccct/panels.py` | 4.6, 5, 6 |
| Mechanism controls and the size check | `ccct/mechanism.py` | 7 |
| Power calibration at panel dimensions | `ccct/powercurve.py` | 7 |
| Interface ablation: the D+ block | `ccct/dplus.py` | 15 |
| Command line entry point | `ccct/run.py` | 11 |
| Independent verifier | `verify_ccct.py` | 12 |

## Commands

```bash
python -m ccct.run identity                                     # replicate zero
python -m ccct.run ablation                                     # section 15
python -m ccct.run panel --panel jonikas --replicates 1000 --placebo 100 --workers 4
python -m ccct.run panel --panel costanzo --replicates 2000 --workers 4
python -m ccct.run control --settings B1 B2 B4 B5 --replicates 500 --workers 4
python -m ccct.run control --settings B4 --datasets 200 --replicates 39 --workers 4
python -m ccct.run power --datasets 40 --replicates 99 --workers 4
python -m ccct.run summarize
python verify_ccct.py --identity
```

## Three things a reader should check first

**The runner is the release.** `ccct/frozen.py` calls `kernel.fit` and
`selection.select_from_inner` with the same arguments, in the same order, as
`nested.train`. `python -m ccct.run identity` asserts that on both panels,
bitwise, in one process, and also that the shared-kernel pass is bitwise
neutral and that the identity permutation reproduces the unpermuted run.
Agreement with the archived CSVs is `2.2e-16` rather than bitwise: the release
is bitwise deterministic across identical fresh processes but shifts by about
`4e-15` under a different module import order, which is a property of the
release's numerics and not of this code.

**The exclusion follows provenance, not names.** Permuting D moves where a
record's measurements came from without moving its name, so the archived
endpoint exclusion no longer bounds what the training set knows about the test
record's D provenance. Every replicate rebuilds the nested structure from the
union of the name-sharing and provenance-sharing graphs; under the identity
permutation this reproduces the archived exclusion set for every outer group,
which replicate zero asserts. On Jonikas the pool goes from 66.3 to about 53
training pairs under permutation. On Costanzo the rule is a verified no-op
under within-fold permutation, which is why `V1_fold` is the primary variant
there and the global permutation is a sensitivity.

**The reference is placebo matched, and the p-value is approximate.** The
denser exclusion shrinks training pools under the null but not for the
untouched observed statistic. The reference is therefore the mean of `K` draws
that keep the real D blocks and apply a permutation's exclusion graph. Two
sources of approximation remain and are named in every output record: records
sharing genes are not exchangeable, and the null and placebo exclusion graphs
are equal in law but not identical.

## The D+ block

`ccct/dplus.py` rebuilds the Jonikas third-partner context from
`data/jonikas/080930a_DM_data.mat`, because the release ships the prepared P/D
interface rather than the historical Jonikas D adapter. The rebuild is not
assumed: the module recomputes the frozen context coordinates 18 to 22 from the
raw arrays and refuses to return a D+ block unless all five match bitwise. That
check passing is what makes the three restored odd descriptors descriptors of
the same measurements rather than a re-derivation of unknown fidelity.

## Output

```
ccct_results/
  ccct_summary.json            collected results, Holm family, decision
  ccct_identity.json           replicate zero
  ccct_ablation.json           section 15
  panels/ccct_<label>.json     one per panel run
  controls/ccct_<setting>.json one per control setting
  power/ccct_power_jonikas_dims.json
  replicates/<label>.csv       every replicate's statistic and selection counts
  ccct_verification.json       written by verify_ccct.py
```
