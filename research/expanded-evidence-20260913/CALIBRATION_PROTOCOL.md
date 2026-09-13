# Secondary score calibration: fixed plan before opening gold

This addendum is frozen while B is still being collected and before any individual gold score of the selected cohort is shown to the evaluator. The primary A/B experiment and its stopping rule remain unchanged. No original conversational prediction may be overwritten.

Question: can a small supervised mapping improve ordinal boundaries of the actual conversational judge scores without changing semantic observations? This is postprocessing of those recorded scores, not another LLM run and not a GigaChat result.

Use leave-one-connected-client-cluster-out cross-fitting. For each of the 16 held-out clusters, fit on all other clusters. Never use the held-out cluster's labels, comments, tags, or source text. Publish predictions for all 96 units and all five fixed profiles; do not select and report only the best one.

1. Train-mode baseline, tie -> lowest score.
2. Monotone mapping of B scores to 0/1/2. Enumerate the 10 nondecreasing mappings of three ordered levels. Select maximal training unweighted kappa; tie -> closest to identity, then lexicographic. Undefined training kappa is not zero. With insufficient training classes use the training mode and record fallback.
3. Multinomial logistic regression, C=1, no class weights, fixed random_state=20260913 and max_iter=1000. Inputs are six fixed one-hot indicators for A and B categories, including absent levels. Return category argmax and expected score as separate outputs.
4. Same logistic regression with class_weight='balanced'.
5. Isotonic regression on B, increasing=True, y_min=0, y_max=2, out_of_bounds='clip'. Round expected score to the nearest allowed category; exact halfway ties go to the lower category.

Report kappa, ordinal/nominal alpha, category Spearman and continuous-score Spearman separately, accuracy, distributions, recall, severe recall, FPR and coverage. Compare each profile with uncalibrated B using 2000 paired cluster bootstrap replicates. These are secondary exploratory comparisons, without familywise-error correction, on fixed out-of-fold predictions (models not refitted within bootstrap). None can independently authorize deployment or replace the primary result.

Sensitivity checks: (a) change only labels of a held-out cluster and confirm that predictions for that cluster do not change; (b) reject single-cluster cross-fitting rather than falling back to in-sample fitting; (c) no transformation is allowed to reconstruct within-category ranking from information not present in A/B. Reproducing marginal label frequencies is not evidence of good item-level judgment.
