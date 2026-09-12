# Pipeline 2 reference artifacts

This tree holds fitted integration and provenance artifacts for the six
legacy candidates, separately from model source packages:

```text
reference_artifacts/
  random_forest/
  xgboost/
  lightgbm/
  catboost/
  tabpfn/
  xtab/
```

These files support loading, compatibility, and sanity checks. They must not
be reused as dataset-specific trained estimators across the TabArena
benchmark; each benchmark dataset requires a fresh training lifecycle.
RealMLP and TabM have no reference artifact; both are freshly trained through
their pytabkit-backed candidate wrappers.

`xtab/official_iter_2k_backbone.ckpt` is the official XTab cross-table
FT-Transformer backbone released by BingzhaoZhu/XTab. It is loaded before
downstream optimization; dataset-specific tokenizers and prediction heads are
initialized fresh. `xtab/xtab.pt` remains only as the staged legacy
wine-quality fitted-artifact compatibility check.
Source: `https://github.com/BingzhaoZhu/XTab/tree/main/pretrained_ckpt/iter_2k`;
SHA-256: `A067CC251C5086D36AC518F1F6D119AAEE41FF58042AB641C58E0DD5EF9154B8`.

Artifact moves into this tree preserve the original binary bytes. Python
pickle and Torch compatibility aliases depend on serialized module names, not
the containing filesystem directory.
