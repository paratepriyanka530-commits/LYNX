# Pipeline 2 Phase C: target audit and CB-CHL experimental capability

## Status

The deterministic target/imbalance audit is complete. A paper-faithful,
binary-only CB-CHL objective is implemented as an optional LightGBM mode, but
no anneal or MIC experiment has been run. Phase C remains on hold for a
research-design decision because neither dataset has a justified cost matrix
and both eligible datasets are multiclass, whereas the source formulation is
binary-only.

## Target-registry resolution

The canonical registry preserves three distinct target concepts. The
`provided_target_column` records the supplied PDF, the
`authoritative_target_column` records OpenML suite 457 metadata, and the
`benchmark_target_column` is the authoritative target used for future TabArena
comparisons. `target_column` remains a compatibility alias of
`benchmark_target_column`.

| Dataset | Provided/PDF target | Authoritative and benchmark target | Resolution |
|---|---|---|---|
| `miami_housing` | `structure_quality` | `SALE_PRC` | `resolved_to_authoritative_target` |
| `physiochemical_protein` | `SpatialDistNK` | `ResidualSize` | `resolved_to_authoritative_target` |

Statistics were recomputed from the benchmark targets. Both records remain
regression tasks with no missing target values. The complete suite remains 51
datasets: 38 classification (30 binary, 8 multiclass) and 13 regression.

## Eligibility

The Phase C1 imbalance eligibility rule is purely statistical:
`minority_class_ratio < 0.01`. It marks two datasets:

- `anneal`: 0.008908685968819599 (5 classes)
- `MIC`: 0.007062978222483814 (8 classes)

Eligibility does not automatically activate CB-CHL and does not establish
method applicability. Both eligible datasets are multiclass, which matters
because the published objective is binary-only.

## CB-CHL source audit

Source: Xiaosong Zhao, Yong Liu, and Qiangfu Zhao, “Improved LightGBM for
Extremely Imbalanced Data and Application to Credit Card Fraud Detection,”
*IEEE Access*, vol. 12, 2024, pp. 159316–159335,
doi: [10.1109/ACCESS.2024.3487212](https://doi.org/10.1109/ACCESS.2024.3487212).
The project copy is
`C:\Users\HP\Downloads\LYNX Research Papers\Improved_LightGBM_for_Extremely_Imbalanced_Data_and_Application_to_Credit_Card_Fraud_Detection.pdf`.

For binary labels `y_i ∈ {0,1}`, raw score `z_i`, probability
`p_i = sigmoid(z_i)`, and signed label `y_i* = 2y_i - 1`, the paper defines
`p_i* = p_i` for a positive row and `1-p_i` for a negative row. Its fixed
squared-difficulty CB-CHL loss (Equation 16) is:

`L = -Σ_i α_i ζ_i (1-p_i*)² log(p_i*)`

The three components are:

- Class balance (Equation 10): `α_i = y_i + (S1/S0)(1-y_i)`, where `S1` and
  `S0` are positive and negative counts in the training labels.
- Hard-example emphasis: `(1-p_i*)²`; the exponent 2 is fixed by the published
  method, not exposed as an invented tuning parameter.
- Cost sensitivity (Equation 12): `ζ_i = C_FN,i y_i + C_FP,i(1-y_i)` using an
  explicit per-instance binary cost matrix.

With `u_i = α_i ζ_i y_i*(1-p_i*)²` and
`v_i = 2p_i*log(p_i*) + p_i* - 1`, Equation 18 gives `g_i = u_i v_i`.
Equations 21–23 give:

`du_i/dp_i* = -2 α_i ζ_i y_i*(1-p_i*)`

`dv_i/dp_i* = 2log(p_i*) + 3`

`h_i = (v_i du_i/dp_i* + u_i dv_i/dp_i*) y_i* p_i*(1-p_i*)`

Algorithm 1 supplies these same gradient/Hessian operations. The objective has
no additional published loss hyperparameter beyond the fixed squared
difficulty term. Ordinary LightGBM tree settings remain the existing LYNX
standard settings.

The paper defines instance-dependent costs for binary credit-card fraud:
false-positive cost is 0.3% of transaction amount and false-negative cost is
the transaction amount. Those domain assumptions do not transfer to anneal or
MIC. The paper does not define a multiclass objective, multiclass gradient or
Hessian, or a multiclass cost tensor/reduction. No such extension is inferred
in LYNX.

OS-CHL is described as applying oversampling to training data before the CHL
objective. It is not implemented in this phase; Phase C2 is limited to CB-CHL.

## Implementation and safety contract

Standard LightGBM remains the canonical candidate and the default
`training_mode="standard"` path retains its existing configuration and fit/
predict behavior. The optional `training_mode="cb_chl"` path requires all of:

1. classification task;
2. `cb_chl_eligible=True`;
3. explicit activation with `training_mode="cb_chl"`;
4. an explicit `(n_training_rows, 2)` matrix ordered as `[C_FP, C_FN]`; and
5. binary 0/1 training labels.

Missing/invalid costs, non-eligible activation, non-binary labels, and cost
shape mismatches raise controlled configuration errors. There is no default
anneal/MIC cost matrix. The numeric costs in tests are labeled `TEST_ONLY` and
are not persisted as research configuration.

The model binds `S1`, `S0`, the balance factor, and the cost matrix only when
`train_model` receives `y_train`. Test labels are not accepted by this API and
cannot contribute to training state. The benchmark runner still defaults to
standard LightGBM; no CB-CHL benchmark result has been generated.

## Validation

Focused tests cover default-standard behavior, explicit activation, missing
cost rejection, eligibility gating, finite and correctly shaped gradient/
Hessian output, numerical finite-difference agreement, train-only state,
multiclass rejection, and deterministic target resolution. Existing LightGBM
lifecycle and full Pipeline 2 tests are run separately as the final validation
gate.

## Remaining research decision

Before either eligible dataset can yield a scientifically defensible CB-CHL
result, the project must supply both (a) a justified dataset-specific cost
design and (b) a cited multiclass extension or an explicitly approved binary
reframing. Until then, performance comparison is intentionally blocked.
