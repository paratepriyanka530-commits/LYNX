# Pipeline 2: Prediction

The ten candidate implementations are peer packages under `models/`:
`random_forest`, `xgboost`, `lightgbm`, `catboost`, `tabpfn`, `xtab`,
`realmlp`, `tabm`, `tabiclv2`, and `ft_transformer`.

Fitted integration/provenance artifacts are stored separately under
`reference_artifacts/<model>/`. They are verification references, not
cross-dataset benchmark estimators.

`models/artifact_io.py` remains at the models root because it is shared by
the model packages. Model-specific helpers remain inside their owning package.

Semantic responsibilities are intentionally separated. Pipeline 1 retains
column-level semantics for safe cleaning decisions. Pipeline 2 uses one holistic
dataset-level semantic profile together with statistical and target/task metadata for
model recommendation. Generate or resume those profiles serially with:

```powershell
C:\Users\HP\.virtualenvs\lynx-pipeline2-eval-py313\Scripts\python.exe -m benchmarks.tabarena.pipeline2_groundwork dataset-semantic --output-dir benchmarks/tabarena/pipeline2_artifacts --model qwen2.5:7b --llm-timeout 600
```

The older `semantic` command remains the detailed Architecture-B column profiler and
is not the selector semantic-input path.
