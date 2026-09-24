# Pipeline 2 selector input freeze

Date: 2026-09-14

## Frozen input contract

The selector joins datasets only by integer `dataset_id` and models only by canonical
`base_model_id`. Filenames and display names are provenance, not join keys. The base
evaluation grid is 51 datasets x 10 canonical models = 510 combinations. Infeasible
combinations remain explicit `infeasible_skip` rows with null performance; they are
never dropped or encoded as zero.

The authoritative inputs are:

- `pipelines/pipeline2_prediction/inputs/dataset_metadata.json`
- `pipelines/pipeline2_prediction/inputs/dataset_semantics.json`
- `pipelines/pipeline2_prediction/inputs/dataset_target_registry.json`
- `pipelines/pipeline2_prediction/inputs/model_capabilities.json`
- `pipelines/pipeline2_prediction/training_data/model_performance.json`

`selector_join_contract.json`, `selector_leakage_audit.json`, and
`selector_input_validation.json` make the join, leakage, and validation rules machine
readable. Rebuild these deterministic artifacts with:

```powershell
C:\Users\HP\.virtualenvs\lynx-pipeline2-eval-py313\Scripts\python.exe -m pipelines.pipeline2_prediction.build_selector_inputs
```

## Collection status

| Input | Coverage | Status |
|---|---:|---|
| Statistical metadata | 51/51 | Complete; recomputed from local CSV predictors after exact target exclusion |
| Semantic profiles | 51/51 | Complete; one validated dataset-level `qwen2.5:7b` request per dataset |
| Target registry | 51/51 | Complete and validated |
| Model capabilities | 10/10 | Complete for the canonical candidate roster |
| Model performance | 0 rows | Intentionally empty schema/template; no benchmark outcomes fabricated |

Pipeline 2 semantics are intentionally holistic. Each profile contains
`probable_domain`, `subdomain`, `dataset_purpose`, `row_entity`, `target_concept`,
`feature_context`, `semantic_summary`, and `domain_confidence`. Its evidence is
compact and deterministic: dataset/task/target identity, dimensions, feature-type
counts, and at most 80 predictor names sampled evenly in source order. The target is
excluded from that sample. No per-column semantic vector is required by the selector.

This establishes the research boundary:

- Pipeline 1: column-level semantics -> safe cleaning decisions.
- Pipeline 2: dataset-level semantics + statistical metadata + target/task metadata
  -> model recommendation.

The old `pipeline2_artifacts/semantic_profiles/` files remain intact for provenance.
The existing `semantic` command remains available, but those column-level artifacts
are no longer selector inputs. Authoritative Pipeline 2 profiles live under
`pipeline2_artifacts/dataset_semantic_profiles/`.

Semantic collection is resumable at dataset granularity: each completed profile is
persisted immediately and existing completed profiles are skipped. Continue it with:

```powershell
C:\Users\HP\.virtualenvs\lynx-pipeline2-eval-py313\Scripts\python.exe -m benchmarks.tabarena.pipeline2_groundwork dataset-semantic --output-dir benchmarks/tabarena/pipeline2_artifacts --model qwen2.5:7b --llm-timeout 600
```

Then rerun the deterministic selector-input builder above.

## Target registry validation

The registry contains exactly 51 datasets: 38 classification and 13 regression;
classification comprises 30 binary and 8 multiclass datasets. The two previously
resolved source/benchmark disagreements remain explicit rather than silently
renamed:

- `miami_housing`: provided `structure_quality`; benchmark target `SALE_PRC`
- `physiochemical_protein`: provided `SpatialDistNK`; benchmark target `ResidualSize`

## Model capability boundary

The canonical candidates are Random Forest, LightGBM, XGBoost, CatBoost, TabPFN,
XTab, RealMLP, TabM, TabICLv2, and FT-Transformer. Hard feasibility limits are kept
separate from empirical operating guidance and provisioning requirements. CRUMB is
an execution strategy for an eligible base model, and CB-CHL is a LightGBM training
mode; neither is an eleventh candidate.

## Leakage boundary

Dataset shape, predictor-only missingness/cardinality/type statistics, holistic
dataset-semantic fields, registered target characteristics, and model capabilities are
inference-available. Held-out predictive and calibration metrics, benchmark runtime,
resource use, failures, OOB outcomes, baseline-relative scores, and oracle ranks are
training outcomes and must appear only in `model_performance.json` after real runs.
The template is deliberately empty until those runs occur.

## Readiness verdict

**GO — SELECTOR INPUTS FROZEN AND BENCHMARK READY**

All 51 required dataset-level semantic profiles are complete. The deterministic
selector-input rebuild validates 51 datasets, 10 models, and the 510-combination base
join without introducing benchmark outcomes.
