# Independent review: pointwise quota selection and bounded differences

## Verdict

The proposed McDiarmid improvement is **valid**, conditional on frozen calibration and training, when each score and contribution is a fixed function of its own input and tie key and the input/key records are independent. It applies to the current exact top-64-of-256 interpolation and histogram policies. Score–gain correlation is unrestricted. The proof does not require labels, bounded scores, independence of score and gain, or independent selections within a batch.

This is a finite-input bounded-difference theorem, not an assertion that the 256 selected/weighted contributions are independent. Ordinary Hoeffding applied to those dependent contributions would be unjustified; the following proof is for the complete selection functional.

## Sharp single-policy sensitivity

Fix a batch of B records, a quota k, and a total priority order determined by pointwise score, tie key, and a fixed index fallback. Let each record's nonnegative weight lie in [0,c]. Let S be the sum of the weights of the k highest-priority records. Replace one record i, including its score, weight and key, leaving all others fixed.

The ordering of all B−1 unchanged records is unchanged. There are four cases:

1. Record i is unselected before and after: S does not change.
2. Record i is selected before and after: the same k−1 other records are selected, and S changes only by its own weight difference, of magnitude at most c.
3. Record i enters the selected set: it replaces the kth highest unchanged record. The two weights both lie in [0,c], so their difference has magnitude at most c.
4. Record i leaves the selected set: the reverse exchange also changes S by at most c.

Thus `|S(new)-S(old)| <= c`. The bound is attained and cannot generally be reduced. Tied scores cause no problem if the total order uses pointwise keys and a fixed fallback, because unchanged records retain their ordering. Independent continuous tie keys make an exact fallback tie a null event; a fixed index fallback also gives the bound for every possible input.

For n independent batches of B=256 inputs, define `F=(1/(Bn))*sum_batches S`. One independent input/key replacement changes F by at most c/(Bn), and there are Bn records. Conditional on the frozen models and policy tables, the sum of squared sensitivities is c²/(Bn).

The Doob martingale for revealing the independent records has conditional increment ranges of width at most their respective sensitivities: integrating over the unrevealed records preserves the replacement bound. Hoeffding's exponential-moment lemma and Chernoff optimization therefore give

`P(F-E[F] >= t | frozen fit) <= exp(-2 Bn t²/c²)`.

Consequently, with probability at least 1−delta,

`E[F] >= observed_F - c sqrt(log(1/delta)/(2*256*n))`.

For the absolute certificate, choose `w(x)=min(J(x),c)`. Since J is nonnegative and the selected mask is exactly half-turn invariant, its expected loss improvement equals E[mask J], which is at least E[mask min(J,c)]. This connects the statistical lower bound to the actual loss improvement. It is not a claim about classification error.

## Two-policy difference and tails

For two frozen pointwise-score quota policies, apply the single-policy bound to each selected sum. Their difference changes by at most 2c. This is sharp even with the same pointwise weights and common tie keys: a weight-c record can be selected only by policy B before replacement and only by policy A afterward, exchanging with weight-zero records. The difference moves from −c to +c.

Take `w(x)=J(x)*1{x in certified region}`, which lies in [0,Jmax] globally. It is harmless that the original J may be unbounded outside the region: the truncated weight is zero there, and the quota is still selected on the original full batch. With fixed calibrated pointwise scores, truncation does not alter the selection rule. The bounded-difference radius for the difference of the two batch means is

`Jmax sqrt(2 log(1/delta)/(256*n))`.

For a full-population relative loss-gain claim, subtract a deterministic upper bound on `E[J*1{outside region}]`. Since the difference of masks is between −1 and 1, this is a valid single tail remainder; policies need not agree outside the region. The bounded two-amplitude and one-amplitude tasks need no tail remainder. The rings certificate separately proves its known-law tail bound. This review does not replace or loosen that bound.

## Conditioning, multiplicity, and limitations

Conditioning on arbitrary learned scores/tables makes this valid for each fitted model without assuming the calibration data are deterministic. Verification inputs and random keys must remain independent of all fitting and selection choices. Once conditioned, shared verification inputs and shared tie keys across the two policies are allowed and desirable for a paired comparison. Reusing inputs across claims and nested sample sizes does not invalidate a union bound.

The proposed joint family has 48 models × 3 claims × 3 looks × 2 methods (empirical Bernstein and McDiarmid) = 864 events. Assigning each delta=0.05/864 gives family error at most 0.05. The maximum of the two simultaneous valid lower bounds is valid on the same joint event. Choosing the first successful prespecified look is also covered by this finite union. No extra data may be adaptively added without an additional valid error allocation. For gain versus the folded policy, retain the same deterministic tail subtraction for both methods.

The sharper theorem would not automatically apply to a selector whose scores depend on the entire verification batch (for example, batch-fitted centers, quantiles used to change scores nonmonotonically, re-estimated policy parameters, or shared random variables that alter all priorities when one record changes). Fixed per-input functions and keys are essential. Merely having an exact quota is not sufficient. The current interpolation and fixed histogram functions meet the needed condition. A monotone ranking transformation that leaves all unchanged priorities intact is harmless, but no such extension is needed here.

## Executable checks and scope

`bounded_difference.py` exhaustively varies binary scores and weights for four-record batches, all nontrivial quotas, tied and reverse-ordered keys, and replacements of the changed key. It also checks 4,096 full 256-record, 64-quota replacements with tied scores/keys and a sharp two-policy example at the actual batch size. `bounded_difference.json` records every failure (if any), counts, maxima, and the proposed numerical radii. No model, fresh state, training sample, or fresh evaluation stream is accessed.

This proof is an independent analytical review of the proposed comparator, not an evaluation of which bound will be tighter on the eventual data. Both methods and all failures must be retained as prespecified.
