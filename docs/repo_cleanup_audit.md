# LYNX Repository Cleanup Audit

**Audit date:** 2026-09-11  
**Mode:** inspection only — no deletion, move, rename, refactor, import edit, or
`.gitignore` change was performed  
**Primary production source:**
`pipelines/pipeline1_cleaning/pipeline.py::run_pipeline1`  
**Pipeline 2 scope:** current groundwork under `benchmarks/tabarena/`

The companion machine-readable classification is
[`docs/repo_cleanup_candidates.json`](repo_cleanup_candidates.json).

## 1. Executive summary

The audit inspected **564 pre-existing files** (approximately 449 MB) across the
repository, excluding `.git/**` internals and excluding the two audit files
created by this task. Git reported 453 tracked files, 27 untracked files, and 84
ignored files. One tracked production file was already modified
(`pipelines/pipeline1_cleaning/pipeline.py`); the frozen guard, Pipeline 2
groundwork, their tests/artifacts, and current research documents are currently
untracked. This report did not alter those states.

### Classification totals

| Category | Files |
|---|---:|
| A. KEEP — ACTIVE PRODUCTION | **143** |
| B. KEEP — TEST / SAFETY COVERAGE | **35** |
| C. KEEP — RESEARCH BASELINE / EXPERIMENT EVIDENCE | **162** |
| D. KEEP — TOOLING / REPRODUCIBILITY | **36** |
| E. REVIEW — POSSIBLY OBSOLETE | **73** |
| F. DELETE CANDIDATE — HIGH CONFIDENCE | **23** |
| G. GENERATED ARTIFACT — DO NOT VERSION | **92** |
| **Total** | **564** |

The repository is not dominated by dead source code. Most volume comes from
downloaded TabArena datasets and repeated generated datasets. The safest cleanup
is therefore staged artifact/cache removal, not source deletion.

The most important caution is `tests/pipeline1_multidomain/results/`: its 45
files are byte-for-byte identical to `results_v001_baseline/`, but it is the
active output/resume path of `run_multidomain_test.py`, and the static dashboard
plus `frontend/tests/reportAdapter.test.js` read reports from it. Do not remove
or rename that tree until these consumers are deliberately redirected.

## 2. Files definitely active

### Frozen Pipeline 1 runtime — 33 Python files

Keep all Python files under `pipelines/pipeline1_cleaning/` except the inactive
Architecture A implementations `semantics/column_understanding.py` and
`semantics/role_resolver.py`, plus all interpreter caches. The active chain is:

```text
pipeline.py::run_pipeline1
  → ingestion/{file_validator,dataset_loader}
  → profiling/{column_profiler,dataset_profiler}
  → quality/{issue_detector,quality_report}
  → semantics/{evidence_builder,dataset_understanding,batched_analyzer,
               batched_schema,semantic_schema,contradiction_guard}
  → decisions/{cleaning_decision_engine,decision_schema}
  → orchestration/{action_registry,action_validator,data_cleaner_performer}
  → techniques/{date_parsing,numeric_coercion,missing_values}
  → validation/post_clean_validator
  → output/clean_dataset_exporter
```

`semantic_schema.py` must remain even though much of it originated with
Architecture A: `batched_schema.py` imports its type literals and
`batched_analyzer.py` imports `DatasetSemanticInference`. Likewise,
`dataset_understanding.py` is partially active because Architecture B imports
its dataset RAG and prompt builders.

Git status within this group: 31 files tracked/clean,
`pipeline.py` tracked/modified, and `semantics/contradiction_guard.py` untracked.

### Intelligence and knowledge dependencies — 98 files

- Eleven production Python files under `intelligence/llm/` and
  `intelligence/rag/` implement the provider abstraction, Ollama adapter,
  knowledge loading/chunking, MiniLM embeddings, in-memory vector search, and
  retrieval. Only the two mock implementations are test-only.
- Eighty-seven Markdown documents under `knowledge_base/` are indexed by
  `RAGRetriever.index_knowledge()`. Root/source README, manifest, and schema
  documents are intentionally excluded from indexing but remain authoring tools.

### Active Pipeline 2 groundwork — 3 files

- `benchmarks/tabarena/pipeline2_groundwork.py` — current 51-task manifest,
  download/sync, statistical meta-feature, frozen semantic-profile, feasibility,
  and model-log groundwork. **Untracked.**
- `pipelines/pipeline2_prediction/models/random_forest/` — the implemented
  model wrapper. Tracked.
- `pipelines/pipeline2_prediction/__init__.py` — package boundary. Tracked.

### Current dashboard and freeze documentation — 9 files

Six frontend runtime assets consume real Pipeline 1 reports, including the image
asset referenced twice by `index.html`. Keep them unless frontend scope is
explicitly retired. Also keep the untracked `FROZEN.md` and two research
architecture specifications: they describe the just-frozen system.

## 3. Tests worth keeping

### Core production and safety regression tests

The following 15 files contain real assertions and cover active behavior or an
intentional historical contract:

- `tests/integration/test_pipeline1_vertical_slice.py`
- `tests/pipeline2/test_groundwork.py`
- `tests/pipeline2/test_semantic_confidence_auditor.py`
- `tests/unit/test_batched_semantic_analyzer.py`
- `tests/unit/test_benchmark_v001.py`
- `tests/unit/test_cleaning_decision_engine.py`
- `tests/unit/test_coerce_dtype.py`
- `tests/unit/test_column_semantic_contract.py`
- `tests/unit/test_impute_missing.py`
- `tests/unit/test_issue_detector.py`
- `tests/unit/test_parse_dates.py`
- `tests/unit/test_pipeline1_vertical_slice.py`
- `tests/unit/test_role_resolver.py`
- `tests/unit/test_semantic_contradiction_guard.py`
- `frontend/tests/reportAdapter.test.js`

The role/column-contract tests cover inactive Architecture A behavior, but that
behavior remains a research baseline and is imported by Benchmark V001. They are
not obsolete merely because the frozen runtime does not call the modules.

Keep the 17 valid/invalid dataset fixture files (excluding the redundant parent
`.gitkeep`), both semantic mock providers, and `tests/__init__.py`. The 13 large
domain CSVs are directly enumerated by the multidomain runner; the generic
business fixture is also used by vertical-slice and live safety tests. The MIC
dataset is ignored under TabArena, but the contradiction-guard test reads it and
therefore requires it locally.

## 4. Research baseline files worth keeping

### Architecture A

Keep `column_understanding.py` and `role_resolver.py` as an explicitly labelled
Architecture A baseline. Reference searches found direct use by:

- `benchmarks/pipeline1_semantics/run_benchmark_v001.py`;
- `benchmarks/pipeline1_semantics/ollama_prompt_cache_diagnostic.py`;
- unit semantic retrieval/contract/role tests;
- four manual integration semantic runners;
- six benchmark result families and current architecture documentation.

Deleting either would break reproducibility and tests even though neither is in
`run_pipeline1`.

### Benchmark and diagnostic evidence

Keep, preferably in a clearly named research-artifact archive:

- 31 Pipeline 1 semantic benchmark/diagnostic result files;
- both 45-file multidomain result trees until canonicalization is approved;
- 15 non-duplicate integration reports/live audits;
- 13 missing-value diagnostic files, including the two untracked GPU results;
- `semantic_understanding_results.json`;
- four Architecture A manual semantic runners;
- the untracked MIC diagnostic/audit CSV+JSON pairs and `task_b_summary.json`;
- the untracked `model_inventory.json`, which records that only one of six
  intended Pipeline 2 model candidates currently exists.

These files are generated, but their paper/reproducibility value outweighs a
blanket “generated means delete” rule.

## 5. Tooling/reproducibility files worth keeping

- `benchmarks/pipeline1_semantics/run_benchmark_v001.py` and
  `ollama_prompt_cache_diagnostic.py`.
- TabArena download/check utilities, low-signal runner, and semantic auditor.
- `pipeline2_artifacts/model_run_log_schema.json`.
- Multidomain runner and result consolidator.
- Five live missing/date/numeric evidence builders/runners.
- Twelve manual local RAG/embedding/Ollama probes. These should eventually move
  from `tests/` to a manual diagnostics/tools directory, but they remain useful.
- `.gitignore`, `.env.example`, `README.md`, `requirements.txt`, and the four
  knowledge-base authoring/provenance documents.

`requirements.txt` currently lists only `PyYAML`, while active code imports at
least pandas, NumPy, Pydantic, requests, sentence-transformers, scikit-learn,
SQLAlchemy/python-dotenv for the dormant DB subsystem, and an Excel engine where
applicable. It is incomplete but must be repaired in a separate task, not
deleted during cleanup.

## 6. Possible obsolete files — manual review required

Every E-category item is listed below. “No references” means `rg` searches were
performed across Python, JavaScript, Markdown, and JSON sources/results, with
large generated report bodies excluded where they could only add embedded
copies.

| Path | Type / Git | Evidence and references | Tests / docs / benchmarks | Research value | Risk | Recommendation |
|---|---|---|---|---|---|---|
| `PROJECT_STRUCTURE.md` | file / tracked | Does not mention the frozen contradiction guard or current Pipeline 2 groundwork; not read at runtime | Documents older tree and Architecture A; no test/runner consumes it | High as historical snapshot | MEDIUM | **ARCHIVE**, then replace separately if desired |
| `app/**` | folder, 3 files / tracked | Welcome-message stub plus empty config; no imports found | Listed in old structure only | Low/future app boundary | MEDIUM | **KEEP** pending product-scope decision |
| `api/**` | folder, 4 files / tracked | Empty package and three `.gitkeep` placeholders; no imports | No tests/benchmarks; planned docs only | Low/future scaffold | MEDIUM | **KEEP** pending scope decision |
| `database/**` except redundant `.gitkeep` | folder, 17 files / tracked | Dormant but internally connected SQLAlchemy subsystem; several zero-byte model stubs | Two environment-bound integration scripts import repositories; old structure documents it | Medium prototype value | HIGH | **ARCHIVE** only after DB decision |
| `mcp/**` | folder, 9 zero-byte/placeholder files / tracked | No imports or implementations found | No tests/benchmarks; planned structure only | Future integration scaffold | MEDIUM | **KEEP** pending scope decision |
| `security/**` | folder, 6 zero-byte/placeholder files / tracked | No imports; actual safety logic lives in Pipeline 1 | No tests/benchmarks; planned structure only | Future scaffold | MEDIUM | **KEEP** pending scope decision |
| `shared/**` | folder, 5 zero-byte Python files / tracked | No imports found | Old structure only | Namespace reservation | MEDIUM | **KEEP** pending scope decision |
| `storage/**` | folder, 4 `.gitkeep` files / tracked | No references; production uses caller-provided output directory | Old structure only | Future storage taxonomy | MEDIUM | **KEEP** pending scope decision |
| `pipeline1_cleaning/techniques/{categorical,duplicates,inconsistencies,missing_values,outliers,text_cleaning,type_conversion}/` | 7 empty folders / tracked `.gitkeep` | Superseded structurally by top-level technique modules and registry; no imports/path uses | Tests import active top-level modules; old structure depicts these folders | Shows early planned taxonomy | MEDIUM | **ARCHIVE** after confirming no planned split |
| `intelligence/llm/prompts/` | empty folder / tracked `.gitkeep` | Production prompts are inline; no path use found | No tests/benchmarks | Low scaffold value | LOW | **ARCHIVE** |
| `docs/{architecture,database,pipeline1,pipeline2,pipeline3}/` | 5 empty folders / tracked `.gitkeep` | No content/references; current architecture is in `docs/research/` | No tests/benchmarks | Intended documentation taxonomy | LOW | **KEEP** until docs layout is chosen |
| `pipelines/pipeline3_results/**` | folder, 2 files / tracked | Empty package and placeholder README; no imports | No tests/benchmarks | Future Pipeline 3 boundary | MEDIUM | **KEEP** pending roadmap decision |
| Eight manual/environment-bound files named `test_*` | files / tracked | No unittest cases. DB pair needs external DB; Random Forest file calls `input()`; five unit scripts hard-code missing `G:\My Drive\...pizza_sales.xlsx` | Not reliable automated coverage; some listed in old docs | Medium as manual experiment code | MEDIUM | **ARCHIVE/MOVE** to manual diagnostics in a later task |
| `intelligence/schemas/__init__.py` | zero-byte file / tracked | No import found; active schemas live under Pipeline 1 | No tests/benchmarks; planned docs only | Future shared schema boundary | MEDIUM | **KEEP** pending architecture decision |

The eight mislocated manual files are:

```text
tests/integration/test_dataset_registration.py
tests/integration/test_profile_storage.py
tests/pipeline2/test_random_forest.py
tests/unit/test_column_profiler.py
tests/unit/test_column_understanding.py
tests/unit/test_dataset_profiler.py
tests/unit/test_semantic_evidence.py
tests/unit/test_semantic_retrieval.py
```

## 7. High-confidence delete candidates

No deletion was performed. These candidates have direct reference and hash
evidence, but should still be reviewed as one change set.

### 7.1 Twelve duplicate integration cleaned datasets

SHA-256 analysis found three exact duplicate groups:

- the three `pipeline1_architecture_b_pizza_output{,_v2,_v3}` cleaned CSVs;
- the five categorical/coerce/impute/date/trim business cleaned CSVs;
- the four decision/repair/issue/vertical-slice business cleaned CSVs.

All are tracked. No code/test opens these files; their paired reports contain
historical output-path strings, but those strings are audit metadata rather than
runtime dependencies. Original input fixtures and the more informative reports
remain. **Risk: LOW. Recommendation: DELETE**, ideally after recording hashes or
packaging the reports as the canonical evidence.

### 7.2 Nine redundant `.gitkeep` files

The following tracked placeholders sit in directory trees that already contain
tracked files, have no references, and no longer preserve an empty directory:

```text
database/repositories/.gitkeep
intelligence/schemas/.gitkeep
knowledge_base/cleaning_methods/.gitkeep
knowledge_base/data_quality/.gitkeep
knowledge_base/domains/.gitkeep
knowledge_base/sources/.gitkeep
tests/datasets/.gitkeep
tests/integration/.gitkeep
tests/unit/.gitkeep
```

**Risk: LOW. Recommendation: DELETE.**

### 7.3 Two unreferenced invalid-format samples

`tests/datasets/invalid/lab DM (Copy).pdf` and
`tests/datasets/invalid/test.txt` have no path references and no current
file-validation test consumes them. The PDF is also copy-named. **Risk: LOW.
Recommendation: DELETE**, unless a future negative-fixture test is first added.

## 8. Generated artifacts / Git-ignore candidates

| Path | Type / Git | Why generated | References and value | Risk | Recommendation |
|---|---|---|---|---|---|
| Ten `**/__pycache__/` folders containing 31 `.pyc` files | folders/files / ignored | Python bytecode caches | No imports/docs/tests/benchmark dependency; recreated automatically | LOW | **DELETE locally**; existing ignore is correct |
| `benchmarks/tabarena/datasets/` | folder, 53 files / ignored | Downloaded OpenML/TabArena CSVs and metadata, about 265 MB | Required locally by active groundwork, download/check tools, MIC calibration, and MIC-backed guard test; reproducible via OpenML and manifests | HIGH before benchmark | **GITIGNORE and KEEP LOCAL** until sweep completes |
| Eight mutable `pipeline2_artifacts` manifest/state/checkpoint files | files/folder / untracked, not ignored | Generated by `pipeline2_groundwork.py` | Fixed paths are read/written by groundwork; semantic checkpoint avoids an expensive rerun; final manifest/features may merit release archive | MEDIUM | **GITIGNORE** mutable state; archive final immutable release separately |

The eight mutable Pipeline 2 items are the dataset manifest CSV/JSON,
`dataset_sync.json`, `groundwork_summary.json`, `model_run_log.csv`,
`semantic_run_status.json`, `statistical_meta_features.json`, and
`semantic_profiles/task_363621.json`. Do not remove the checkpoint before the
planned semantic sweep decision.

The MIC diagnostic/audit files and `task_b_summary.json` are deliberately not in
this generated-ignore bucket: they are current research evidence and should be
archived/versioned according to the paper reproducibility policy.

## 9. Duplicate/superseded file groups

| Group | Active/current member | Duplicate or superseded member | Evidence | Disposition |
|---|---|---|---|---|
| Multidomain results | `tests/pipeline1_multidomain/results/` | `results_v001_baseline/` | All 45 corresponding files have identical SHA-256 hashes | Keep both for now; pick a canonical archival tree only after redirecting runner/dashboard/test references |
| Pizza integration outputs | Paired JSON reports | Three identical cleaned CSVs across original/v2/v3 | Same SHA-256; reports differ and retain experiment evolution | Delete duplicate CSVs; archive reports |
| Business integration outputs | Paired JSON/live audit evidence | Five identical cleaned CSVs in transformation output dirs | Same SHA-256; original dataset fixture remains | Delete duplicate CSVs; retain reports/live audits |
| Business no-op outputs | Paired JSON reports | Four identical cleaned CSVs in decision/repair/issue/vertical dirs | Same SHA-256; no code reads CSV copies | Delete duplicate CSVs; retain reports |
| Column semantics | `batched_analyzer.py` + `batched_schema.py` | `column_understanding.py` | Frozen `run_pipeline1` imports batched analyzer; old benchmark/tests import standalone module | Keep old module as Architecture A research baseline, not production |
| Structural roles | Architecture B prompt/default `semantic_role=unknown` | `role_resolver.py` | No frozen production import; Benchmark V001 and role tests import it | Keep as Architecture A baseline |
| Dataset semantics | Architecture B calls selected helpers from `dataset_understanding.py` | `understand_dataset()` full standalone path | Batched analyzer imports retrieval/prompt helpers and `DatasetSemanticInference` | Do not delete or split without refactoring; partially active hidden dependency |
| Test vs runner naming | Real unittest files | Eight non-asserting/interactive/environment-bound `test_*.py` files | Static scan found no test cases; five hard-coded G-drive paths, one calls `input()` | Move/archive later; do not count as automated tests |
| TabArena acquisition | `pipeline2_groundwork.py` contains manifest sync | `download_tabarena.py`, `download_missing.py`, `check_missing_tasks.py` | Overlapping download/check responsibilities, but older utilities may recover/verify datasets independently | Keep tooling pending a command-by-command equivalence review |
| Repository documentation | Frozen architecture docs | `PROJECT_STRUCTURE.md` | Old document omits current guard/groundwork and describes inactive architecture as current | Archive old snapshot; do not silently overwrite historical context |

No duplicate action registry, post-clean validator, or current decision-engine
implementation was found. The apparent old/new semantic duplication is a
research architecture split with live benchmark references, not dead code.

## 10. Recommended cleanup order

1. **Preserve current state first.** Decide how the modified/untracked frozen
   Pipeline 1 and active Pipeline 2 files will be reviewed and committed. Cleanup
   must not accidentally discard them.
2. **Remove ignored interpreter caches locally.** This is independent, reversible
   through regeneration, and zero research risk.
3. **Delete redundant `.gitkeep` files** in already populated trees as one small,
   reviewable change.
4. **Canonicalize integration evidence.** Record hashes, retain reports/live
   audits, then remove the 12 byte-duplicate cleaned CSVs.
5. **Choose a multidomain canonical tree.** Before touching either 45-file tree,
   redirect `run_multidomain_test.py`, `consolidate_results.py`,
   `frontend/datasets.js`, and `frontend/tests/reportAdapter.test.js`; verify the
   dashboard and tests; only then remove the redundant copy.
6. **Define an artifact policy.** Separate immutable paper evidence from mutable
   run state; add only the latter to `.gitignore`. Keep ignored TabArena datasets
   locally until the planned sweep is complete.
7. **Move manual runners out of `tests/`.** Preserve history while making test
   discovery reliable; replace valuable probes with assertions where warranted.
8. **Archive Architecture A and stale structure docs explicitly.** Do not delete
   them while Benchmark V001 remains a reproducibility claim.
9. **Review dormant scaffolds by roadmap owner.** Handle app/API/DB/MCP/security/
   shared/storage/Pipeline 3 together so namespace decisions are intentional.
10. **Repair reproducibility metadata separately.** Expand dependency management
    and document commands only after cleanup paths stabilize.

## Final ranked lists

### Top 10 safest deletion candidates

1. All ten ignored `__pycache__/` directories (31 `.pyc` files).
2. `database/repositories/.gitkeep`.
3. `intelligence/schemas/.gitkeep`.
4. `knowledge_base/cleaning_methods/.gitkeep`.
5. `knowledge_base/data_quality/.gitkeep`.
6. `knowledge_base/domains/.gitkeep`.
7. `knowledge_base/sources/.gitkeep`.
8. `tests/datasets/.gitkeep`.
9. `tests/integration/.gitkeep` and `tests/unit/.gitkeep`.
10. The 12 byte-duplicate integration `cleaned_dataset.csv` files, after recording
    hashes/accepting that paired reports retain historical path strings.

The two invalid-extension fixtures rank immediately after these because no
consumer was found, but retaining tiny negative fixtures costs little.

### Top 10 files/groups that look obsolete but should not be deleted yet

1. `semantics/column_understanding.py` — inactive production, active Architecture
   A benchmark/test baseline.
2. `semantics/role_resolver.py` — inactive production, active role benchmark/tests.
3. `semantics/dataset_understanding.py` — standalone path looks old, but
   Architecture B imports its retrieval and prompt helpers.
4. `semantics/semantic_schema.py` — Architecture B imports its literals and
   dataset inference contract.
5. `tests/pipeline1_multidomain/results/` — byte-duplicate, but runner,
   consolidator, dashboard, and frontend test depend on it.
6. `tests/pipeline1_multidomain/results_v001_baseline/` — byte-duplicate, but it is
   the explicitly named research baseline.
7. `PROJECT_STRUCTURE.md` — stale as current docs, valuable as a historical map.
8. The eight mislocated manual `test_*.py` runners — poor tests but potentially
   useful experiment tools.
9. `database/**` — dormant from Pipeline 1, yet internally coherent and used by
   two integration scripts.
10. `benchmarks/tabarena/datasets/` and the semantic checkpoint — generated and
    ignored/untracked, but currently needed to avoid redownload/re-inference.

## Surprising dependencies and repository risks

- **Hidden frontend dependency:** `frontend/datasets.js` reads dataset 09 and 13
  reports from the nominally generated multidomain `results/` tree, and the Node
  adapter test reads the same tree.
- **Partial legacy-module dependency:** Architecture B does not call
  `understand_dataset()`, but imports two functions from its module; deleting the
  “old” file would break production.
- **Shared-schema dependency:** the Architecture A-looking semantic schema remains
  transitive production code.
- **Freeze is not yet represented by a clean Git state:** the guard and freeze
  docs are untracked while `pipeline.py` is modified. This is not a cleanup target.
- **Active Pipeline 2 is also untracked:** groundwork, tests, inventory, and all
  artifacts are outside the index; datasets alone are correctly ignored.
- **Test discovery pollution:** eight `test_*.py` files are actually manual
  scripts; five fail on a machine-specific G-drive path and one prompts for input.
- **Dependency manifest gap:** `requirements.txt` is far smaller than the active
  import surface, so a clean clone is not currently reproducible from it alone.
- **Absolute-path provenance:** many historical `cleaning_report.json` files embed
  old absolute output paths. This is useful provenance but can be stale and may
  expose workstation layout in published artifacts.
- **Model inventory gap:** Pipeline 2 currently has one implemented candidate
  where six are expected; do not mistake empty model space for safe scaffold
  deletion.

No repository cleanup was performed.
