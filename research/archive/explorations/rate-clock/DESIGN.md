# Does a smaller step merely delay the same behavior?

The earlier comparison kept the number of GD updates fixed at 400. At rates 0.5, 1 and 2 this gives different values of the nominal gradient-flow time, the learning rate times the number of GD updates. This follow-up uses the same 32 trained starting states and evaluation draws as the prior study. Both seed groups have already been used to estimate fixed-update effects. The new contrasts are follow-ups on those models.

## Comparisons

1. From the same 100-update pullback state, train full-batch GD at 0.5, 1 and 2. The main comparison is at nominal time 800, corresponding to the earlier endpoint of 400 GD updates at rate 2. This gives the smaller rates 1,600 and 800 GD updates respectively. Report both the number of seeds preferring the region rule and the mean within-seed loss difference. Report earlier times and nominal time 1,600 as a persistence check. The scalar time is a gradient-flow approximation and does not measure the actual distance traveled by a model.
2. Compare prediction rules when the training loss reaches common levels along the different paths, including the training loss achieved in each seed by GD at rate 2 after 400 updates. Select checkpoints using only training information. Report the number of comparable seeds and achieved loss differences. The rule being assessed uses 64 second evaluations per batch of 256. Positive region loss minus confidence loss favors confidence.
3. Evaluate the Hessian of the training loss at the shared switch state and record actual losses on the first 25 GD updates at each rate. Compare the local quadratic stability scale with observed early oscillations. This local test concerns the first updates. It does not identify the mechanism causing the final routing preferences.

The study reports the original and second groups separately where informative and uses paired resampling of complete seed vectors for group contrasts. The direction of an effect, its size and whether the routing winner changes are separate outcomes. None of these finite-time checks establishes that two models converge to different limits.
