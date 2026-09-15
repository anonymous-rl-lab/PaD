# Section 15 result: the interface ablation

Draft text for the paper. Numbers are from `ccct_results/ccct_ablation.json`
and `ccct_results/controls/ccct_ablation_mechanism.json`, protocol
`b76a942bae0d1acab9be8a1f81a8effdc7c083c4bfddada02e7e872f18dcfa8b`.

## The finding in one sentence

The mechanism's raw ordered profile is a noise-free direction oracle, exactly
as Supplementary S4.2 says; restoring it on the real panel yields descriptors
at chance. The exchange-invariant context interface is therefore not a device
that manufactures difficulty, and we can now say so with numbers instead of
with a design choice.

## Main real-data table, additional row block (Jonikas)

| Learner | D coordinates | AUROC | AP |
|---|---|---|---|
| PaD(D) | 23 | 0.881114 | 0.4154 |
| A_match(D) | 23 | 0.827664 | 0.3281 |
| A_rbf(D) | 23 | 0.810982 | 0.3014 |
| PaD(D+) | 26 | 0.866213 | 0.4230 |
| A_match(D+) | 26 | 0.585358 | 0.1541 |
| A_rbf(D+) | 26 | 0.748785 | 0.2803 |

`D+` adds three odd context descriptors to the retained block: the mean
partner-wise residual difference, its signed rank fraction, and the mean raw
profile difference, each computed from exactly the inputs that produce the
retained context coordinates 20 to 22 and each flipping sign under endpoint
exchange. Their panel-level scaling constants follow the rule the existing odd
descriptors use and are frozen before any fit. The rebuild of the raw
third-partner context is validated by requiring that it reproduce the frozen
context coordinates 18 to 22 bitwise.

Contrasts: `Delta_1 = 0.2958`, `Delta_2 = 0.2809`, `Delta_3 = -0.2423`.

## Text

Restoring the discarded direction does not let the additive control close the
gap; it opens it (`Delta_1 = +0.296`). The Jonikas joint-minus-additive gain is
therefore not an artifact of the retained representation having deleted
directional information.

The reason is that the restored direction carries almost none on this panel.
Measured directly against the reference annotations, with a 84-pair bootstrap
interval, the mean partner-wise residual difference reaches AUROC 0.517
[0.415, 0.623] and its signed rank fraction 0.512 [0.373, 0.654], both at
chance; the mean raw profile difference reaches 0.368 [0.299, 0.444], that is
weakly anti-aligned with the reference convention rather than aligned with it.
For scale, the frozen P score alone reaches 0.536 [0.397, 0.672] on the same
panel.

The mechanism behaves in the opposite way, and we report it. On B1 with the
same three descriptors restored, the matched additive control falls from a
0.2949 error rate to **0.0000** over 4,096 independent test instances, at all
three prescribed seeds. In the mechanism the ordered profile pair is
`(0, B e_W)` or `(B e_W, 0)` with `B > 0`, so a signed partner difference
recovers the direction exactly. That oracle is a property of the generator.
It is not present in the Jonikas assay.

We therefore state the design choice as an empirical result. The
exchange-invariant context interface is what makes the mechanism's D channel
background-only, and on the real panel the directional information it discards
is not measurable at this panel size. The abstract accordingly says that a
*retained phenotype representation* can carry no directional information on its
own, not that an assay can.

## What we do not claim

`Delta_2 = 0.281` is not evidence that synergy survives the interface, and we
do not present it that way. It is inflated by the additive control degrading,
not by PaD improving: three near-chance odd coordinates become three of eleven
in the linear D component and carry about 1.9 times the mean variance of the
existing 23 coordinates in the RBF mean squared distance, and the additive
class has no way to discount them. Its selection moves to the pure D-odd kernel
in 36 of 84 outer groups, from 6, and to the strongest regularizer in 31 of 84,
from 7 -- the signature of a learner absorbing added noise. PaD(D+) moves only
from 0.881 to 0.866, because a product kernel can suppress an uninformative D
factor.

For the same reason `Delta_3 = -0.242` should not be read as "the discarded
context carried a large amount of usable direction, with the wrong sign". It
mixes the question of whether restored direction helps with the question of
whether three noisy coordinates hurt. The per-coordinate measurement above is
the clean answer to the first question, and it is the one we report.

## Pre-registration status

Readings R1 to R4 were fixed in protocol v1.1 before execution. As they fall:
R1 is false (`Delta_1 = 0.296 > 0.01`), R3's criterion is not met
(`|Delta_3| = 0.242 > 0.01`), and R2's first clause holds (`Delta_2 >= 0.03`)
but its second clause is decided by the CCCT on D+. Formally this is R4. The
per-coordinate AUROC measurement that establishes R3's substance was not
pre-registered and is reported as a post-hoc diagnostic.
