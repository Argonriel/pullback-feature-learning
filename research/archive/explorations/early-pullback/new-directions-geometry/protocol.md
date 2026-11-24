# Direction and feature-step length

This local development experiment fixes its design before evaluating the new validation pairs. It asks whether scaling the ordinary gradient step to the predicted feature-step length of pullback reproduces its early classification performance. A reciprocal control asks what happens when pullback uses the ordinary gradient's predicted length.

Use eight paired two-moons training and validation sets with 256 observations per split, noise 0.15, and the existing 80-parameter network and fixed readout. For pair i = 1,...,8, use seeds 61000+i for training, 62000+i for initialization, and 63000+i for validation. Generate only training and validation splits. The optimizer settings are GD rate 0.125, pullback rate 8, and damping 0.1. Keep these settings fixed.

At each arm's own current state, compute candidate parameter increments dG and dP using ordinary gradient descent and damped pullback with their stated rates. For training Jacobian J and training count n define r(d) = ||Jd||_2 / sqrt(n). Run four arms through 500 updates:

| Arm | Increment |
| --- | --- |
| gd | dG |
| pb | dP |
| gd_at_pb_size | dG * r(dP) / r(dG) |
| pb_at_gd_size | dP * r(dG) / r(dP) |

A zero target yields a zero increment. An exactly zero source with positive target makes the normalization undefined and stops the experiment. Record numerical failures without substituting another seed, rate, or arm. Compare predicted lengths on every rescaled update with an absolute tolerance of 1e-12 plus 1e-10 times the target length. Track the actual finite movement, measured with the same RMS norm on representation differences. The matching target depends on the arm's own state. Arms need not share a cumulative distance once they diverge.

Prespecified main endpoints are validation logistic loss and fixed-readout accuracy after 100 and 500 updates. Also record training metrics, cumulative actual and predicted movement, validation differences within seeds, and steps 0, 1, 10, and 50 for context. These experimental controls require both directions, so computational timing is outside the question.

To directly compare directions at identical states, branch one hypothetical update in both directions from the shared initialization and the pure GD and pure pullback states before updates 100 and 500. For each state compare both candidate directions using the same predicted length, first with the GD length and then with the pullback length. Record immediate actual movement, training and validation changes, and directional alignment with the desired training feature velocity. These diagnostic branches do not change the four evolving arms.
