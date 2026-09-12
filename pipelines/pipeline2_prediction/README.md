# Pipeline 2: Prediction

The eight candidate implementations are peer packages under `models/`:
`random_forest`, `xgboost`, `lightgbm`, `catboost`, `tabpfn`, and
`xtab`, `realmlp`, and `tabm`.

Fitted integration/provenance artifacts are stored separately under
`reference_artifacts/<model>/`. They are verification references, not
cross-dataset benchmark estimators.

`models/artifact_io.py` remains at the models root because it is shared by
the model packages. Model-specific helpers remain inside their owning package.
