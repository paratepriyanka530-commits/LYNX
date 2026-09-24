# Pipeline 2 Phase E — calibration and reliability metrics

Status: benchmark collection support implemented; no full benchmark run.

## Probability capability audit

All 10 canonical wrappers provide class predictions through the common
`predict_model` path. Their fitted classification lifecycles also provide
probabilities and a verifiable probability-column order:

| Model | Probability path | Fitted class order |
|---|---|---|
| Random Forest | sklearn `Pipeline.predict_proba` | `Pipeline.classes_` |
| XGBoost | `LYNXXGBoost.predict_probability`; binary matrix reconstructed | inner `XGBClassifier.classes_` |
| LightGBM | `LYNXLightGBM.predict_probability`; binary matrix reconstructed | inner `LGBMClassifier.classes_` |
| CatBoost | `LYNXCatBoost.predict_probability`; binary matrix reconstructed | inner `CatBoostClassifier.classes_` |
| XTab | `XTabCandidate.predict_proba` | fitted target mapping ordered by encoded index |
| TabPFN V3 | `LYNXTabPFN.predict_probability` | inner `TabPFNClassifier.classes_` |
| TabM-D | shared `PytabkitCandidate.predict_proba` | fitted estimator `classes_` |
| TabICLv2 | shared `PytabkitCandidate.predict_proba` | fitted estimator `classes_` |
| RealMLP-TD | shared `PytabkitCandidate.predict_proba` | fitted estimator `classes_` |
| FT-Transformer | shared `PytabkitCandidate.predict_proba` | fitted estimator `classes_` |

This is wrapper-level support, not an inference from library documentation.
Classification probability capability metadata is stored in both
`MODEL_CAPABILITIES` and `model_inventory.json`.

CRUMB remains an experimental execution strategy rather than a canonical
model. Its current cluster-aggregation wrapper does not expose a probability
path, so CRUMB calibration is recorded as unavailable. No base-model
probability is fabricated or inferred from hard predictions.

## Calibration definitions

Binary Brier score uses the second fitted class as the positive class and is
the sample mean of `(p_positive - 1[y=positive])^2`.

Multiclass Brier score is the sample mean of the sum across classes of
`(p_class - one_hot_target_class)^2`. It is **not** divided by the number of
classes. This convention is frozen because published multiclass
normalizations differ.

ECE is top-label, equal-width ECE with exactly 15 bins. For each sample,
confidence is the maximum predicted probability and correctness indicates
whether its argmax class equals the true class. Bins are left-closed and
right-open—`[0,1/15)`, ..., `[14/15,1]`—with confidence exactly `1` assigned to
the final bin. Every non-empty bin contributes its sample fraction multiplied
by the absolute difference between mean correctness and mean confidence.

Stored constants are `ece_n_bins=15` and
`ece_method="top_label_equal_width"`; the bin count is not tuned.

## Probability validation and failure behavior

Before Brier, ECE, ROC-AUC, or log-loss is accepted, the evaluator requires:

- a two-dimensional probability matrix;
- one row per test target;
- one column per fitted expected class;
- finite values within `1e-6` of `[0,1]`;
- row sums within absolute tolerance `1e-6` of one;
- unique, known class ordering matching the train-fitted classes; and
- exact alignment of every test label to that ordering.

Values within boundary tolerance are clipped only after validation. Any
probability retrieval or validation failure is isolated to that run's
probability-dependent outcomes: Brier, ECE, ROC-AUC, and log-loss become null,
and `status` plus the exact reason are logged. Accuracy and other hard-label
metrics remain available. The benchmark loop does not fail solely because a
probability metric is unavailable.

## Additional predictive metrics

Classification records now contain accuracy, balanced accuracy, macro
precision/recall/F1/F2, weighted precision/recall/F1, Matthews correlation
coefficient, and a confusion matrix. Binary ROC-AUC uses the second fitted
class's probability. Multiclass ROC-AUC uses macro one-vs-rest. Log-loss uses
the validated probability matrix and explicit fitted class labels. The frozen
primary metric remains balanced accuracy and is still stored in the original
`metric_name`/`metric_value` fields.

Regression records contain RMSE, MAE, R², and NRMSE. NRMSE is RMSE divided by
the population standard deviation of `y_test`; it is null when test-target
variance is zero. The frozen primary RMSE and baseline-relative score remain
unchanged.

Ordinary point-prediction regression has no predictive distribution or
interval in the current protocol. Consequently regression calibration is
`not_applicable`, with Brier/ECE null. Point residual metrics are not described
as calibration and no uncertainty is synthesized.

## Schema and leakage boundary

Model-run schema version 5 adds JSON-serializable `predictive_metrics`,
`calibration_metrics`, and `probability_metadata` blocks while retaining
`base_model_id` and `execution_strategy`. In-memory legacy records receive
explicit defaults during validation so prior result structures remain
readable and serializable.

Brier, ECE, ROC-AUC, log-loss, and all other test-set scores are benchmark
outcomes only. They must never enter semantic/statistical meta-features or any
inference-time selector input because they are unavailable for a new unseen
dataset. They may be analyzed later as outcomes or objectives.

## Validation and limitations

Deterministic tests cover binary and multiclass Brier/ECE, exact bin count and
boundaries, invalid matrices, unavailable probabilities, regression handling,
failure isolation, schema compatibility, CRUMB metadata, and small Random
Forest binary/multiclass/regression lifecycles with serialization-compatible
metric structures. Smoke results establish plumbing only and support no model
performance conclusion.

Limitations: calibration is measured on the same held-out test split used for
performance reporting; no confidence intervals, adaptive-bin ECE, calibration
curves, or post-hoc recalibration are introduced. ECE is a finite-bin summary
and should not be interpreted as a complete characterization of calibration.
