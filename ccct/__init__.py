"""Cross-Channel Correspondence Test (CCCT).

A prospectively specified permutation test for whether the joint-minus-additive
gain of PaD exceeds what the same joint learner produces on data in which the
two channels carry no correspondence beyond what each carries about the label.

The package adds no new data and refits nothing outside the frozen procedure of
the v4 release. See protocols/CCCT_FROZEN_PROTOCOL.md.
"""
__all__ = ["frozen", "operators", "panels", "mechanism", "run"]
