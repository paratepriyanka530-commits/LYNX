# Pipeline 2 Phase D — CRUMB context-selection integration

Status: optional experimental strategy implemented; no benchmark sweep run.

## Source audit

The implementation is grounded in Heredge et al., *CRUMB: Efficient Prior
Fitted Network Inference via Distributionally Matched Context Batching*,
arXiv:2606.11473v1 (9 June 2026). The local reviewed PDF has SHA-256
`577646246B3272B31C3D953F0A86CE491EE11805C58C344FE3D32BFF5CB99250`.
No official implementation repository was linked by the paper or found in its
official TeX source, so the equations and pseudocode are the implementation
authority.

The paper explicitly evaluates TabPFNv2, TabICLv1, and TabICLv2. It addresses
the quadratic training-context attention cost of prior-fitted networks without
retraining or changing their weights. Algorithm 1 is:

1. Standardize train and test features.
2. Cluster test queries into `K` groups using Euclidean k-means, Lloyd's
   algorithm, and k-means++ initialization.
3. For each test cluster, greedily select `n` training points that minimize the
   Gaussian-RBF MMD objective between that cluster and its context.
4. Run one base-PFN prediction for each `(selected context, test cluster)` pair
   and restore predictions to original test-row order.

Candidate score at greedy step `t` is the paper's relevance/redundancy score:
`-2 mean(k(candidate, test cluster)) + 2/(t+1)
sum(k(candidate, selected))`. Subsets for different clusters may overlap.
The accelerated implementation uses the source defaults `K=20`, 64 random
Fourier features, and greedy batch size `B=50`. Gaussian bandwidth uses the
median heuristic. For bounded auxiliary memory, LYNX computes that median on a
fixed-seed sample of at most 2,048 combined standardized train/test rows; this
sampling bound is an explicit engineering detail because the paper does not
specify how to compute the median at large `N`. The paper's optional large-data
early stopping is implemented: inspect MMD every 50 selected points after
`max(100,n/10)` warm-up and stop after five consecutive relative improvements
below `1e-4`.

The main experiments use context size `n=0.1N`; large-data context is capped at
the base model window. The paper assumes numeric vectors in `R^d`, requires the
entire test-feature batch for clustering, and performs CPU clustering/context
selection followed by `K` PFN forward passes. Its stated limitations include
full-test-batch availability and Euclidean k-means/RBF weakness on mixed or very
high-dimensional data.

## Compatibility and design

- `tabiclv2_crumb` is directly source-supported. It wraps the unchanged merged
  TabICLv2 interface and uses the paper's 4,096 context cap locally. It does not
  create a 4,096-row hard gate for standard TabICLv2.
- `tabpfn_crumb` is interface-compatible but not directly validated by the
  paper: the paper used TabPFNv2, while LYNX currently uses TabPFN V3. It wraps
  the unchanged TabPFN package and preserves its classification-only, 500
  feature, 10-class, and 10,000-context restrictions.
- Both variants expose the common four-function interface. They are registered
  under `EXPERIMENTAL_EXECUTION_STRATEGIES`, not `EVALUATION_MODEL_IDS`.
  Therefore the canonical selector pool remains exactly 10 models.
- The runner includes variants only through explicit repeated
  `--experimental-strategy` options. Rows record the distinct variant
  `model_id`, `base_model_id`, and `execution_strategy=crumb`; standard rows
  record `execution_strategy=standard`.
- The base feasibility result is retained. `crumb_feasibility.json` separately
  records `base_model_feasible`, `crumb_variant_feasible`, context size, and
  whether the strategy changes feasibility.

## Leakage and preprocessing boundary

The runner splits before target encoding or model work. CRUMB's `fit` accepts
only `X_train,y_train`; `predict` accepts only `X_test`, so a test-label input
does not exist. Test features are intentionally used for test clustering, the
median heuristic, and MMD relevance, exactly as the method requires. The CRUMB
scaler is fitted on training features only and only transforms test features.
Selected labels are indexed exclusively from `y_train`.

The source does not prescribe categorical encoding or missing-value handling
for selection distances. LYNX therefore rejects non-numeric or non-finite
selector matrices with a controlled error instead of silently imposing an
unsourced distance representation. This restriction concerns only the optional
strategy and does not change either base wrapper's preprocessing.

## Current 51-dataset feasibility audit

The canonical matrix remains 510 rows (51 datasets × 10 models). Its current
TabPFN count remains 21 feasible and 30 infeasible. Separate CRUMB analysis has
102 rows (51 × two strategies):

- TabPFN: CRUMB changes the deterministic row/context gate for 15 datasets.
  The other 15 base-infeasible datasets remain blocked by task type (13
  regression) or feature count (2 above 500). The 15 row-gate changes are
  `Amazon_employee_access`, `APSFailure`, `bank-marketing`,
  `credit_card_clients_default`, `customer_satisfaction_in_airline`,
  `Diabetes130US`, `E-CommereShippingData`, `GiveMeSomeCredit`, `heloc`,
  `HR_Analytics_Job_Change_of_Data_Scientists`,
  `in_vehicle_coupon_recommendation`, `kddcup09_appetency`,
  `online_shoppers_intention`, `SDSS17`, and `jm1`.
- TabICLv2: zero feasibility changes because the current verified base profile
  has no hard maximum-row gate; all 51 base combinations are already feasible.

The 15 figure is specifically the theoretical hard-gate change requested by
Phase D. Actual variant execution additionally requires a finite numeric
selector matrix. It must not be presented as 15 completed runs.

## Controlled validation

A synthetic 40-row train / 6-row test lifecycle used a deliberately constrained
7-row context cap and three test clusters. It verified deterministic valid
indices, exactly seven context rows per base-model call, correct six-row output
shape and original ordering, train-only scaler statistics, and exclusive use of
training labels. This is an execution/safety smoke test, not a performance
result. No real benchmark model was trained and no accuracy claim is made.

Focused coverage lives in `tests/pipeline2/test_crumb.py`, alongside updated
feasibility and runner-schema coverage. Generated metadata is in
`benchmarks/tabarena/pipeline2_artifacts/crumb_feasibility.json`.

## Limitations

1. TabPFN V3 compatibility is an architecture-level extrapolation; CRUMB's
   paper tested TabPFNv2, not V3.
2. No official CRUMB code was available for byte-for-byte behavior comparison.
3. Mixed/categorical or missing selector features require a separately
   source-grounded representation before execution; they currently fail
   explicitly.
4. The bounded median sample is documented implementation machinery absent
   from the paper's reproducibility details.
5. CRUMB requires an available batch of test features and performs multiple
   base-PFN fits/forward passes; runtime and memory were not benchmarked here.
6. Feature-only MMD selection does not guarantee that every classification
   context contains every target class; a base model may reject a selected
   single-class context. LYNX does not add unsourced label-aware selection.
