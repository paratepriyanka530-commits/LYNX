# Pipeline 2 six-model verification

Date: 2026-09-11

Scope: focused pre-benchmark verification only. No large-scale dataset
evaluation was run.

## 1. Random Forest identical-prediction investigation

The earlier check passed the following synthetic row twice:

```text
pizza_id=1, order_id=1, pizza_name_id="test", quantity=1,
order_date="2015-01-01", order_time="12:00:00", total_price=12.0,
pizza_size="M", pizza_category="Classic",
pizza_ingredients="Cheese", pizza_name="Test Pizza"
```

The two inputs were not different. Identical predictions therefore provided
no evidence of degeneracy.

The loaded artifact is
`pipelines/pipeline2_prediction/reference_artifacts/random_forest/random_forest.pkl`,
SHA-256
`5722050709F0CFA3E8ED81BA86B9BD38E7E7940C14D8F67FB542BDD90C04B765`.
It is a fitted sklearn `Pipeline` containing a 200-tree
`RandomForestRegressor`. Its metadata declares target `unit_price` and
task `regression`. Consequently, `12.0` was a continuous regression
output, not an encoded class.

## 2. Random Forest stronger sanity result

Five real rows were selected from `pizza_sales.csv` across distinct pizza
types, sizes, categories, dates, total prices, and target-price quantiles:

| Row index | pizza_name_id | quantity | total_price | size | category | Prediction |
|---:|---|---:|---:|---|---|---:|
| 41785 | pepperoni_s | 1 | 9.75 | S | Classic | 9.75 |
| 10 | bbq_ckn_s | 1 | 12.75 | S | Chicken | 12.75 |
| 48544 | ital_supr_m | 1 | 16.50 | M | Supreme | 16.50 |
| 20432 | veggie_veg_l | 1 | 20.25 | L | Veggie | 20.25 |
| 19441 | the_greek_xxl | 1 | 35.95 | XXL | Classic | 35.94999999999988 |

Result: shape `(5,)`, five distinct predictions, minimum `9.75`, maximum
`35.94999999999988`, and all values finite. The artifact is not
prediction-degenerate. The near-direct relationship between `total_price`,
`quantity`, and `unit_price` makes this an artifact/data-design caveat,
not evidence of unusually broad predictive performance.

## 3. XTab environment provenance

Checkpoint inspection found three top-level entries:
`model_state_dict`, `model_config`, and `preprocessor`. It contains no
Torch version, Python version, general package manifest, or training-runtime
metadata. The pickled `StandardScaler` state embeds scikit-learn `1.6.1`.
PyTorch archive values `format_version=1`, `version=3`,
`storage_alignment=64`, `byteorder=little`, and the serialization ID
describe the archive format and are not the training Torch version.

The staged requirement `torch==2.8.0` was reproduced in a separate CPython
3.13 environment while holding NumPy `2.1.3`, pandas `2.2.3`, and
scikit-learn `1.6.1` constant.

| Runtime | Predictions | Probability shape |
|---|---|---|
| Torch 2.11.0+cu128, CPU inference | 5, 5, 5 | `(3, 6)` |
| Torch 2.8.0+cpu | 5, 5, 5 | `(3, 6)` |

Maximum absolute probability difference: `5.960464477539063e-08`.
This is negligible floating-point variation. Compatibility is demonstrated
for the known input under both versions, but the original
**training-runtime provenance remains unresolved**.

## 4. XTab legacy alias mechanism

The checkpoint references
`model.xtab_preprocessor.XTabPreprocessor`. The production loader
temporarily maps:

- `model` to `pipelines.pipeline2_prediction.models.xtab`
- `model.xtab_preprocessor` to
  `pipelines.pipeline2_prediction.models.xtab.xtab_preprocessor`

`XTabPredictor.load` applies these aliases automatically around every
`torch.load`. The context manager records and restores the prior
`sys.modules` entries in a `finally` block. Tests confirm loading with no
manual setup, restoration when the names were initially absent, restoration
of pre-existing sentinel modules, and two consecutive loads with no
persistent alias.

Sequential repeated loads are safe. Separate benchmark processes are safe
because each process has its own `sys.modules`. Concurrent threads loading
the checkpoint could interleave changes to the process-global module table;
thread-parallel model loading is therefore not certified.

## 5. Six-model feasibility matrix

Limits below describe the merged implementation and supplied artifacts, not
the broader capabilities of the underlying libraries. `unknown` means no
deterministic restriction was found.

| Model | Tasks | Min rows | Max rows | Max features | Max classes | Binary / multiclass | Missing values | Categorical / numeric |
|---|---|---:|---:|---:|---:|---|---|---|
| Random Forest | classification, regression | unknown | unknown | unknown | unknown | yes / yes | No wrapper prohibition; numeric-null artifact behavior not separately certified | One-hot categorical; numeric supported |
| XTab | classification, regression | 10 | unknown | unknown | unknown | yes / yes | Numeric means; learned missing/unknown categorical tokens; regression target must be complete numeric | Both supported |
| LightGBM | classification only | 2 effective | unknown | unknown | 2 | yes / no | Fitted numeric medians; categorical null token | Label-encoded categorical with unseen-label rejection; numeric supported |
| XGBoost | classification only | 2 effective | unknown | unknown | 2 | yes / no | Training fills nulls; inference fill state is not persisted | Label-encoded categorical with unseen-label rejection; numeric supported |
| CatBoost | classification only | 1 explicitly checked | unknown | unknown | 2 | yes / no | Fitted numeric medians; categorical null token | Object/string categorical and int64/float64 numeric |
| TabPFN | classification only | 2 effective | 10,000 | 500 | 10 | yes / yes | Passed unchanged; no repository imputation | Repository supplies no encoding or categorical indices; behavior delegated to TabPFN |

Restriction sources:

- Random Forest: common wrapper, fitted sklearn pipeline, and
  `random_forest_metadata.pkl`. The supplied artifact additionally requires
  its exact 11-column pizza schema and was verified under sklearn 1.9.0 /
  joblib 1.6.0.
- XTab: trainer's explicit 10-row check, preprocessor transformations, dynamic
  output dimension, and checkpoint inspection. No hard maxima were found.
- LightGBM: wrapper rejection of regression, binary objective, at-least-two-
  classes validation, and fitted preprocessor.
- XGBoost: wrapper rejection of regression, `binary:logistic` objective,
  at-least-two-classes validation, and fitted preprocessor.
- CatBoost: wrapper rejection of regression, `Logloss` configuration, and
  fitted preprocessor. The wrapper only explicitly checks nonempty training
  data; no stronger row minimum was invented.
- TabPFN: classification-only wrapper, two-class validation, and TabPFN 8.5
  inference configuration (`MAX_NUMBER_OF_SAMPLES=10000`,
  `MAX_NUMBER_OF_FEATURES=500`, `MAX_NUMBER_OF_CLASSES=10`).

The deterministic gate now evaluates task type, XTab's minimum rows, TabPFN's
registered row/feature/class limits, and the binary-only three-model class
limit. It emits one record for each of the six candidates.

## 6. LightGBM and CatBoost second-dataset sanity

The additional real dataset was TabArena `diabetes.csv`: 768 rows, eight
numeric predictors, and a binary diabetes target. This is meaningfully
different from the blood-donation training dataset.

Both saved preprocessors require exactly:

- `MonthsSinceLastDonation`
- `NumberOfDonations`
- `TotalBloodDonated`
- `MonthsSinceFirstDonation`

Both correctly rejected the diabetes schema with a deterministic
`ValueError` listing all four missing columns. Renaming unrelated diabetes
features to bypass this check would not be a valid sanity test and was not
done. Therefore neither existing artifact can produce a legitimate prediction
or probability result on this second domain.

Control checks on seven schema-compatible held-out blood rows succeeded:

| Model | Prediction shape | Probability shape | Finite | Probability row sums | Warnings | Preprocessor reused |
|---|---|---|---|---|---:|---|
| LightGBM | `(7,)` | `(7, 2)` | yes | all 1.0 | 0 | yes |
| CatBoost | `(7,)` | `(7, 2)` | yes | all 1.0 | 0 | yes |

LightGBM uses persisted numeric medians and fitted label encoders. CatBoost
uses persisted numeric medians and retains categorical strings for native
CatBoost handling. The current blood artifacts contain numeric columns only.

## 7. Remaining known limitations

1. XTab's original training runtime is unknown despite equivalent Torch
   2.8/2.11 known-input behavior.
2. XTab's temporary aliases are process-safe but not certified for concurrent
   threaded loads.
3. The supplied LightGBM and CatBoost artifacts are dataset/schema-specific
   and cannot serve as reusable cross-dataset benchmark artifacts.
4. Random Forest's artifact target has a near-direct arithmetic relationship
   with an input feature, so its excellent sanity outputs do not establish
   general predictive quality.
5. XGBoost's current preprocessor does not persist inference-time numeric fill
   values and rejects unseen categorical labels.

## 8. Prior decision (superseded by lifecycle verification below)

**HOLD — do not start the 45/51-dataset evaluation yet.**

Exact blocker: the required second-dataset LightGBM and CatBoost artifact
sanity cannot pass because both artifacts are bound to the blood-donation
schema. Before a large run, the benchmark execution contract must explicitly
use per-dataset train/fitted-preprocessor/predict cycles (rather than reuse
these blood artifacts), and that lifecycle should receive one small
multi-model smoke run on a second dataset.

Status:

- RANDOM FOREST: healthy
- XTAB: healthy with documented limitation
- FEASIBILITY GATE: complete
- LIGHTGBM SECOND SANITY: fail (artifact schema mismatch)
- CATBOOST SECOND SANITY: fail (artifact schema mismatch)
- FINAL: HOLD

## 9. Per-Dataset Training Lifecycle Verification

### Benchmark architecture

The current benchmark groundwork is not an executable dataset-by-model
runner. Its available commands prepare/refresh the manifest, synchronize
datasets, compute statistical/semantic metadata, produce feasibility records,
initialize the model-run log, and summarize progress. It does not import the
six model packages, construct models, split evaluation datasets, train models,
generate predictions, or calculate benchmark metrics. It also does not load
the demonstration artifacts.

There is consequently no current benchmark runtime code path that can prove
the required eight-stage lifecycle. The intended path must still be
implemented or approved:

```text
manifest target
  -> raw train/test split
  -> new dataset-specific preprocessor
  -> fit preprocessor on raw train only
  -> transform raw train and test with that same object
  -> new model instance
  -> fit model on transformed train only
  -> predict/probability on transformed test
  -> metric and model-run-log record
```

The existing model-specific trainers are not a unified benchmark substitute:

- LightGBM and CatBoost split before fitting their feature preprocessors, but
  fit their target `LabelEncoder` before the split and overwrite fixed
  module-level artifact filenames.
- XGBoost fits its feature preprocessor on the complete feature dataset before
  splitting, which is a direct train/test leakage path. It also writes fixed
  artifact filenames.
- Random Forest's fitted sklearn pipeline is leakage-safe if the benchmark
  passes only its outer training split to `train_model`.
- XTab performs its own train/validation/test split inside the outer training
  input and fits preprocessing on its internal training subset.
- TabPFN passes features directly to the underlying classifier and has no
  repository-level fitted feature preprocessor.

Result: **BENCHMARK ARCHITECTURE — blocker found**. This is an absence/leakage
finding, not evidence that the benchmark reuses old artifacts.

### Diabetes lifecycle smoke test

Dataset: OpenML/TabArena task 363629, `diabetes.csv`, supervised
classification. The authoritative manifest target is
`TestedPositiveForDiabetes`. The dataset has 768 rows and eight numeric
features. A deterministic stratified 80/20 split produced 614 training rows
and 154 test rows.

The smoke test deliberately did not invoke the artifact-writing trainers.
For each model it constructed a fresh preprocessor and model in memory. The
target encoder was fitted on training labels only.

| Check | LightGBM | CatBoost |
|---|---:|---:|
| Task | classification | classification |
| Target | TestedPositiveForDiabetes | TestedPositiveForDiabetes |
| Train/test rows | 614 / 154 | 614 / 154 |
| Original/transformed features | 8 / 8 | 8 / 8 |
| Prediction shape | `(154,)` | `(154,)` |
| Unique predictions | 2 | 2 |
| Probability shape | `(154, 2)` | `(154, 2)` |
| Probability minimum | 0.0000000921387606 | 0.0004523472038592 |
| Probability maximum | 0.9999999078612394 | 0.9995476527961408 |
| Rows sum to approximately 1 | yes | yes |
| Accuracy | 0.7532467532 | 0.7727272727 |
| Runtime warnings/errors | none | none |

Result: **LIGHTGBM FRESH LIFECYCLE — pass** and
**CATBOOST FRESH LIFECYCLE — pass**.

### Leakage evidence

- The target was removed before splitting and was absent from all feature
  frames.
- Raw train and test index sets were disjoint.
- Only the 614-row raw training frame was passed to `fit_transform`.
- The 154-row test frame was passed only to `transform`.
- Persisted numeric fill values exactly equaled medians calculated from the
  training split.
- Diabetes contains no categorical feature columns; consequently both fitted
  categorical mapping sets were empty. The train-only target encoder
  transformed test labels without refitting.
- Preprocessor state snapshots before and after test transformation were
  identical.
- Transformed train and test column layouts were identical: eight columns in
  the same order.
- A new model object was created for each lifecycle and only transformed
  training features were passed to `model.train`.
- Prediction and probability calculation did not mutate or refit the
  preprocessors.

Result: **DATA LEAKAGE CHECK — pass for the controlled smoke lifecycle**.

### Role of saved artifacts

All six checked-in fitted artifacts are classified as **B: integration/sanity
reference artifacts** for benchmark purposes:

| Model | Artifact role | Reason |
|---|---|---|
| Random Forest | B | Fitted pizza/unit-price provenance and integration reference |
| XTab | B | Fitted wine-quality checkpoint used for loader/version verification |
| LightGBM | B | Fitted blood-donation lifecycle reference |
| XGBoost | B | Imported fitted-model/preprocessor provenance reference |
| CatBoost | B | Fitted blood-donation lifecycle reference |
| TabPFN | B | Fitted wrapper artifact reference; fresh construction uses the official pretrained base model |

Some legacy standalone predictor classes default to these paths, but no
benchmark code references them. They must not be reused as the trained
dataset-specific estimator in TabArena runs.

### Six-model fresh-train capability

| Model | Fresh build/train/predict capability | Exception |
|---|---|---|
| Random Forest | yes | Benchmark must make the outer split before fitting its pipeline |
| XTab | yes | Internally subdivides the supplied outer training set |
| LightGBM | yes | Existing trainer's target encoder is fitted before split and artifacts use fixed output paths |
| XGBoost | model interface: yes; existing trainer: noncompliant | Feature preprocessor is fitted before train/test split |
| CatBoost | yes | Existing trainer's target encoder is fitted before split and artifacts use fixed output paths |
| TabPFN | yes | No repository feature preprocessing; preserve feasibility limits and pretrained-weight requirements |

Result: **SIX-MODEL FRESH-TRAIN CAPABILITY — incomplete as an executable
benchmark lifecycle**. All models can be freshly constructed and fitted, but
the XGBoost trainer is leakage-prone and no unified safe runner currently
orchestrates them.

### Final lifecycle decision

- BENCHMARK ARCHITECTURE: blocker found
- LIGHTGBM FRESH LIFECYCLE: pass
- CATBOOST FRESH LIFECYCLE: pass
- DATA LEAKAGE CHECK: pass
- SIX-MODEL FRESH-TRAIN CAPABILITY: incomplete; XGBoost trainer exception
- PIPELINE 2 TESTS: 22/22 passed
- FINAL DECISION: **HOLD**

Exact blockers before the large evaluation:

1. No executable dataset-by-model benchmark lifecycle currently exists.
2. The existing XGBoost trainer fits feature preprocessing before splitting
   and cannot be used as the leakage-safe benchmark path.
3. Existing LightGBM, XGBoost, and CatBoost trainers write fixed artifact
   paths, which would overwrite prior dataset state during a multi-dataset run.

No large benchmark was started.

## Blocker closeout and executable-runner dry run (2026-09-12)

### 1. XGBoost leakage correction

`XGBoostTrainer.train` now splits the raw feature frame before constructing or
fitting `XGBoostPreprocessor`. Only the 614-row diabetes training partition is
passed to `fit_transform`; the disjoint 154-row held-out partition is passed
only to `transform`. The fitted preprocessor persists the training feature
schema and training-only numeric medians. Its state was unchanged by held-out
transformation and prediction, and inference did not call `fit_transform`.

The fresh diabetes lifecycle excluded `TestedPositiveForDiabetes` from all
feature frames, produced predictions with shape `(154,)`, probabilities with
shape `(154, 2)`, row sums exactly 1.0, and accuracy 0.7597402597. Result:
**XGBOOST LEAKAGE FIX — pass**.

### 2. Target-encoder finding

The LightGBM and CatBoost target encoders are sklearn `LabelEncoder`
instances. Fitting one learns only the sorted set of target labels; it does not
consume feature values, label frequencies, target marginals, or any
feature-target relationship. Fitting this vocabulary before the split is
therefore not statistical train/test leakage and cannot tune the fitted model
to held-out outcomes. It can expose the existence of a test-only class, but the
current stratified classification split preserves every eligible class in the
training partition. Per the task's instruction not to make an unnecessary
change when no leakage exists, this code was left as-is. The executable runner
is stricter and fits its target vocabulary on training labels only.

Result: **TARGET ENCODER FINDING — not leakage; left as-is**.

### 3. Per-dataset artifact isolation

The standalone LightGBM, XGBoost, CatBoost, and TabPFN trainers no longer
overwrite fixed reference filenames. Their optional `dataset_id`, `run_id`,
and `artifact_root` inputs resolve paths in the form
`<root>/<dataset_id>/<run_id>/<model_id>/...`. Random Forest had no automatic
trainer save, and XTab already required a caller-provided save path, so neither
had the fixed-write defect. The runner nevertheless applies the same isolated
directory policy to artifacts from all six models. Checked-in reference
artifacts remain unchanged and are not used as fitted dataset estimators; for
TabPFN, only a copy of the immutable pretrained architecture/configuration is
used to construct each fresh estimator, never the reference artifact's fitted
dataset context.

Result: **ARTIFACT FILENAME FIX — fixed for LightGBM, XGBoost, CatBoost, and
TabPFN; not needed for Random Forest and XTab**.

### 4. Executable runner

`benchmarks/tabarena/pipeline2_benchmark_runner.py` now executes the resolved
manifest as dataset × six-model rows. It normalizes task type, makes one fresh
outer split, fits feature preprocessing on training rows only, transforms held-
out rows without refitting, instantiates and trains a fresh model, computes
balanced accuracy for classification or RMSE for regression, computes a
baseline-relative score, measures training and inference separately, and
writes every success, deterministic infeasibility skip, or genuine failure.
Training runtime includes fitted preprocessing, model construction, and fit;
inference runtime includes held-out transformation and prediction.

The log schema is version 2 and includes `task_type`, explicit `status` and
`outcome_reason`, `baseline_relative_score`, separate runtimes, and
`split_seed`. Statistical meta-features are present for both dry-run datasets.
Semantic meta-features are represented in every row and were null here because
these two datasets did not yet have persisted semantic profiles; the runner
loads them by dataset id when the approved semantic-profile artifacts exist.
The CLI requires explicit `--dataset` selectors unless `--all` is deliberately
supplied, guarding against an accidental full sweep.

### Two-dataset × six-model dry run

The final run used split seed 42 and explicit selectors only:

- task 363629, `diabetes.csv`: binary classification, 768 rows, 8 features;
- task 363612, `airfoil_self_noise.csv`: regression, 1,503 rows, 5 features.

XTab used `epochs=1, patience=1` only for this lifecycle smoke run; normal
runner defaults were not changed. All 12 expected rows were logged: eight
successful evaluations and four correct task-type infeasibility skips, with
zero failures and zero dropped rows.

| Dataset | Model | Status | Metric | Value | Baseline-relative | Train s | Inference s |
|---|---|---|---|---:|---:|---:|---:|
| airfoil (363612) | random_forest | success | RMSE | 1.9360 | 0.7245 | 0.2973 | 0.0455 |
| airfoil (363612) | lightgbm | infeasible_skip | — | — | — | — | — |
| airfoil (363612) | xgboost | infeasible_skip | — | — | — | — | — |
| airfoil (363612) | catboost | infeasible_skip | — | — | — | — | — |
| airfoil (363612) | tabpfn | infeasible_skip | — | — | — | — | — |
| airfoil (363612) | xtab | success | RMSE | 5.3837 | 0.2340 | 3.8176 | 0.0045 |
| diabetes (363629) | random_forest | success | balanced accuracy | 0.6965 | 0.3930 | 0.4190 | 0.0503 |
| diabetes (363629) | lightgbm | success | balanced accuracy | 0.6993 | 0.3985 | 0.4417 | 0.0042 |
| diabetes (363629) | xgboost | success | balanced accuracy | 0.6957 | 0.3915 | 0.1534 | 0.0060 |
| diabetes (363629) | catboost | success | balanced accuracy | 0.7185 | 0.4370 | 0.7557 | 0.0022 |
| diabetes (363629) | tabpfn | success | balanced accuracy | 0.7007 | 0.4015 | 0.7066 | 1.1602 |
| diabetes (363629) | xtab | success | balanced accuracy | 0.6615 | 0.3230 | 0.2741 | 0.0048 |

The complete machine-readable rows are in
`benchmarks/tabarena/pipeline2_artifacts/dry_run_blocker_fixes_verified/model_run_rows.json`
and the schema-valid table is in the adjacent `model_run_log.csv`.

Final blocker status:

- XGBOOST LEAKAGE FIX: **pass**
- TARGET ENCODER FINDING: **not leakage; left as-is**
- ARTIFACT FILENAME FIX: **fixed for LightGBM, XGBoost, CatBoost, TabPFN; not needed for Random Forest, XTab**
- RUNNER BUILT: **yes**
- SMALL-SCALE DRY RUN: **pass — 12/12 rows logged, 8 success, 4 infeasible-skip, 0 failure**
- PIPELINE 2 TESTS: **27/27 passed** (plus 21/21 relevant tests in the pinned evaluation environment)
- FINAL DECISION: **GO for the full evaluation after its semantic-profile inputs are populated; the full evaluation was not started**

## TabPFN fresh-checkout verification - pre-fix diagnosis (2026-09-13)

### Installed runtime and current LYNX path

The pinned evaluation environment contains **tabpfn 8.5.0**. The common LYNX
**tabpfn.build_model()** path does not currently pass **model_path=auto**.
Instead, it unconditionally opens the locally ignored
**pipelines/pipeline2_prediction/reference_artifacts/tabpfn/tabpfn_model.pkl**,
copies **models_[0]**, **configs_[0]**, and **inference_config_** into
**ClassifierModelSpecs**, and passes those in-memory specifications to a fresh
**TabPFNClassifier**.

An isolated missing-reference test reproduced fresh-checkout behavior without
moving the retained file: **build_model(classification)** failed immediately
with **FileNotFoundError** from **artifact_io.load_pickle_artifact**. No TabPFN
cache lookup, network request, or authentication flow was reached. Current
local success therefore depends on the retained reference pickle, not on an
existing TabPFN checkpoint cache or authentication token. At verification
time, the reference pickle existed, while the default v3 cache file,
**TABPFN_TOKEN**, **HF_TOKEN**, both TabPFN token files, and the Hugging Face
token cache were all absent.

### What model_path=auto does in TabPFN 8.5.0

When invoked directly, TabPFN treats **auto** as no explicit path and resolves
the configured default **ModelVersion.V3** classifier:

- repository: **Prior-Labs/tabpfn_3**;
- checkpoint: **tabpfn-v3-classifier-v3_default.ckpt**;
- checkpoint size reported by Hugging Face: 212,804,803 bytes;
- Windows default model cache: **%APPDATA%/tabpfn**, unless
  **TABPFN_MODEL_CACHE_DIR** overrides it.

If the checkpoint is missing, TabPFN enters its download path with
**download_if_not_exists=True**. Version 3 is license-gated by TabPFN before
the Hugging Face download call: the package checks **TABPFN_TOKEN**, then
**~/.cache/tabpfn/auth_token**, then **~/.tabpfn/token**; without a usable token
it attempts PriorLabs browser login and license acceptance.

The empty-cache test isolated **TABPFN_MODEL_CACHE_DIR**, **HF_HOME**, **HOME**,
**USERPROFILE**, **APPDATA**, **TEMP**, and **TMP**; removed all TabPFN and
Hugging Face token environment variables in the subprocess; and set
**TABPFN_NO_BROWSER=1**. The package logged that it was beginning the download
workflow for the v3 checkpoint, then raised:

~~~text
tabpfn.errors.TabPFNLicenseError: TabPFN requires a one-time license
acceptance to download model weights for local inference, but browser login
is disabled (TABPFN_NO_BROWSER is set).

Set the TABPFN_TOKEN environment variable with a valid API key obtained from
https://ux.priorlabs.ai
~~~

No checkpoint bytes were written to the isolated cache. A separate isolated
probe with no Hugging Face token successfully resolved the official checkpoint
metadata and fetched bytes 0-1023 with HTTP 206. Therefore Hugging Face
authentication is not required for this checkpoint at verification time;
an **HF_TOKEN** is optional for higher rate limits. The mandatory first-use
gate is PriorLabs login/license acceptance and a valid **TABPFN_TOKEN**.

### Pre-fix clean-machine workaround (superseded)

1. Open https://ux.priorlabs.ai, register or sign in, accept the TabPFN v3
   license on the Licenses page, and copy the API key from the account page.
2. In the shell used for provisioning, set:

   ~~~powershell
   $env:TABPFN_TOKEN = '<your-api-key>'
   ~~~

3. From the repository root, download/cache the official model and regenerate
   the ignored compatibility pickle expected by the pre-fix LYNX wrapper.

   ~~~python
   from pathlib import Path
   import pandas as pd
   from pipelines.pipeline2_prediction.models.tabpfn.tabpfn_model import LYNXTabPFN

   path = Path('pipelines/pipeline2_prediction/reference_artifacts/tabpfn/tabpfn_model.pkl')
   path.parent.mkdir(parents=True, exist_ok=True)
   model = LYNXTabPFN(device='cpu', model_path='auto')
   model.train(pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]}), pd.Series([0, 1, 0, 1]))
   model.save_model(str(path))
   ~~~

Run that snippet with the pinned evaluation Python after setting the token.
It downloads the official v3 checkpoint through TabPFN's package-managed
cache and created the local ignored pickle required by the pre-fix common
interface. The pickle must not be committed.

Pre-fix fresh-checkout verdict: **B. FRESH CHECKOUT REQUIRES ONE-TIME PROVISIONING**.
The behavior was resolved and reproducible, but the pre-fix LYNX common
interface did not fall back to **model_path=auto** by itself when the
reference pickle is absent. Changing that behavior was explicitly outside
this diagnostic-only task.

## TabPFN clean-checkout path fix (2026-09-13)

The production common-interface builder now constructs
**LYNXTabPFN(model_path="auto")** directly. It no longer imports
**ClassifierModelSpecs**, opens the ignored reference pickle, or copies model
state from that pickle. The retained local reference artifact may still be
used for historical diagnostics, but it is not on the production build path.

TabPFN 8.5.0 performs model resolution lazily in **fit()**, not in estimator
construction, and exposes no public eager-download method. Consequently,
**build_model()** now succeeds independently of the reference pickle and the
official cache/download step occurs on the first **train_model()** call:

- Cached weights: the official **model_path="auto"** loader reuses the local
  TabPFN cache; no reference pickle is read.
- Empty cache plus valid **TABPFN_TOKEN**: TabPFN performs its supported
  PriorLabs provisioning and checkpoint download, then training continues.
- Empty cache without valid provisioning: LYNX raises a controlled
  **TabPFNProvisioningError**. Its message states that one-time PriorLabs
  license acceptance is required, browser login may be used when interactive,
  and headless execution requires **TABPFN_TOKEN**. It also states explicitly
  that no reference-pickle fallback was attempted.

The LYNX error contains no credential value and production code neither reads
nor records token contents. Network/auth behavior is mocked in unit tests, so
the tests do not alter the user's cache or credentials.

Focused validation in the pinned Python 3.13 / TabPFN 8.5.0 environment:

- **tests.pipeline2.test_tabpfn_local**: 7/7 passed.
- **tests.pipeline2.test_groundwork**: 9/9 passed.
- The four common functions remain callable.
- The classification-only restriction and limits of 10,000 training rows,
  500 features, and 10 classes remain unchanged.
- No non-TabPFN model wrapper was modified by this fix.

Final verdict: **GO - TABPFN CLEAN-CHECKOUT PATH FIXED**.
