"""Same-rule single-channel controls (SCC).

Three ICLR reviews independently named the same missing control: no D-only
learner fitted under the same rule and the same selector. The frozen P score is
an inherited scalar and the untrained phenotype probe is not a learner, so
neither answers it.

This package fits five learner-panel cells under the release's frozen learning
rule with only the kernel family changed, each capacity-matched to the existing
controls at 35 candidates. It produces no p-value: there is no null
distribution, no permutation and no exchangeability assumption, so there is no
calibration that can fail.

Specification: protocols/SCC_SINGLE_CHANNEL_SPEC.md.
"""
__all__ = ["families", "run"]
