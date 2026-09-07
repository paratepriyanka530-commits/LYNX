# LYNX Project Structure and Responsibilities

## Project purpose

LYNX is an intelligent, preservation-first data-cleaning and prediction project.
Its most developed component is Pipeline 1, which validates and loads a raw
dataset, profiles it, detects quality issues, infers batched semantics, makes
auditable cleaning decisions, compiles allow-listed instructions, executes them
deterministically on a copy, validates the result, and exports a cleaned dataset
plus a JSON report.

The current Pipeline 1 execution path is:

```text
Raw dataset
  -> file validation and ingestion
  -> deterministic profiling
  -> deterministic issue detection
  -> Architecture B batched semantic analysis
  -> deterministic Tier 1 / structured Tier 2 decisions
  -> action-specific authorization and instruction compilation
  -> allow-listed execution on a DataFrame copy
  -> post-clean validation
  -> cleaned_dataset.csv + cleaning_report.json
```

Pipeline 2 and Pipeline 3 currently contain only early scaffolding.

## Top-level hierarchy

```text
LYNX/
|-- api/                         API package placeholder
|-- app/                         Application entry point and configuration
|-- benchmarks/                  Pipeline 1 semantic benchmark programs/results
|-- database/                    SQLAlchemy models and repositories
|-- intelligence/                LLM and retrieval infrastructure
|-- knowledge_base/              Markdown knowledge used by RAG
|-- mcp/                         MCP client/server scaffolding
|-- pipelines/
|   |-- pipeline1_cleaning/      Current end-to-end cleaning pipeline
|   |-- pipeline2_prediction/    Prediction-stage scaffold
|   `-- pipeline3_results/       Results-stage scaffold
|-- security/                    Security package placeholder
|-- shared/                      Cross-project constants, errors, logging, utilities
|-- tests/                       Unit, integration, benchmark, and dataset fixtures
|-- README.md                    Short project introduction
|-- PROJECT_STRUCTURE.md         This architecture and file-responsibility guide
`-- requirements.txt             Python dependencies
```

## `app/`

```text
app/
|-- __init__.py
|-- config.py
`-- main.py
```

- `main.py` is the current minimal command-line entry point. It prints the LYNX
  welcome message and does not yet expose the cleaning pipeline as an API.
- `config.py` is reserved for application settings.
- `__init__.py` marks the directory as a Python package.

## `api/`, `mcp/`, and `security/`

```text
api/
`-- __init__.py

mcp/
|-- __init__.py
|-- client/
|   |-- __init__.py
|   `-- mcp_client.py
`-- servers/
    `-- __init__.py

security/
`-- __init__.py
```

- `api/` is reserved for a future HTTP/API layer.
- `mcp/client/mcp_client.py` is the current Model Context Protocol client
  scaffold; `mcp/servers/` is reserved for server implementations.
- `security/` is reserved for shared security controls.

## `pipelines/pipeline1_cleaning/`

### Pipeline coordinator

```text
pipeline1_cleaning/
|-- __init__.py
`-- pipeline.py
```

- `pipeline.py` implements `run_pipeline1`, the end-to-end coordinator. It
  protects the raw input path, calls every Pipeline 1 stage in order, handles
  action-specific authorization, executes compiled instructions on a copy,
  validates the output, and returns a structured `Pipeline1Report`.
- The default semantic implementation used here is Architecture B through
  `analyze_batched_semantics`.
- The default local LLM is currently `qwen2.5:3b` through Ollama.

### Ingestion

```text
ingestion/
|-- __init__.py
|-- file_validator.py
`-- dataset_loader.py
```

- `file_validator.py` rejects unsupported or invalid input before loading.
- `dataset_loader.py` loads supported dataset formats into a pandas DataFrame
  and returns associated file information.

### Profiling

```text
profiling/
|-- __init__.py
|-- column_profiler.py
`-- dataset_profiler.py
```

- `column_profiler.py` derives deterministic per-column observations such as
  physical dtype, missingness, cardinality, value examples, and numeric/text
  characteristics.
- `dataset_profiler.py` aggregates column profiles and dataset-level facts such
  as dimensions and duplicate-row counts.

### Quality detection

```text
quality/
|-- __init__.py
|-- issue_detector.py
`-- quality_report.py
```

- `issue_detector.py` converts observed profile/data conditions into typed
  `DetectedIssue` candidates. Detection reports evidence; it does not directly
  mutate data.
- `quality_report.py` defines issue contracts and builds aggregate issue
  summaries.

### Semantic understanding

```text
semantics/
|-- __init__.py
|-- batched_analyzer.py
|-- batched_schema.py
|-- evidence_builder.py
|-- column_understanding.py
|-- dataset_understanding.py
|-- role_resolver.py
`-- semantic_schema.py
```

- `batched_analyzer.py` is the active Architecture B semantic path. It combines
  dataset/column evidence with RAG context and asks one structured LLM call for
  a compact dataset context plus semantics for all columns.
- `batched_schema.py` defines Architecture B contracts, including semantic type,
  value function, numeric meaning, descriptions, confidence, and reasoning.
- `evidence_builder.py` turns deterministic profiles into bounded evidence for
  semantic prompts.
- `dataset_understanding.py`, `column_understanding.py`, and `role_resolver.py`
  implement the earlier Architecture A multi-stage semantic path. They are kept
  for benchmarks and regression work but are not restored into the current
  Pipeline 1 Architecture B execution path.
- `semantic_schema.py` contains the Architecture A semantic contracts and
  structural-role vocabulary.

### Cleaning decisions

```text
decisions/
|-- __init__.py
|-- decision_schema.py
`-- cleaning_decision_engine.py
```

- `decision_schema.py` defines the durable contracts for cleaning decisions,
  compiled instructions, execution results, validation reports, artifacts, and
  the final Pipeline 1 report.
- `cleaning_decision_engine.py` implements the hybrid decision engine:

  1. deterministic Tier 1 resolves clear preservation or safe-action cases;
  2. unresolved issues are grouped into bounded Tier 2 structured LLM batches;
  3. missing-value issues use their strict issue-specific response contract;
  4. each returned issue decision is matched and validated independently;
  5. valid siblings are retained immediately;
  6. only missing, duplicated, invalid, timed-out, or otherwise unresolved issue
     IDs are retried in smaller batches;
  7. fallback is applied per unresolved issue;
  8. recovery and rejection details are recorded for auditability.

  The LLM recommends an outcome but does not execute code or directly grant
  final cleaning permission.

### Action orchestration

```text
orchestration/
|-- __init__.py
|-- action_registry.py
|-- action_validator.py
`-- data_cleaner_performer.py
```

- `action_registry.py` is the allow-list connecting stable action IDs to known
  deterministic implementations. Unknown action IDs are rejected.
- `action_validator.py` compiles an approved `CleaningDecision` into a validated
  `CleaningInstruction` and applies action-specific preconditions.
- `data_cleaner_performer.py` executes one instruction deterministically on a
  DataFrame copy and records an `ExecutionResult`.

### Cleaning techniques

```text
techniques/
|-- __init__.py
|-- date_parsing.py
|-- missing_values.py
`-- numeric_coercion.py
```

- `date_parsing.py` assesses date parseability and performs conservative date,
  time, or datetime normalization.
- `numeric_coercion.py` assesses safe numeric conversion and performs coercion
  only when every observed non-null value satisfies the action contract.
- `missing_values.py` defines missingness bands, median mechanical validation,
  the domain-neutral median semantic-eligibility assessor, and deterministic
  imputation preparation/execution helpers.
- Whitespace trimming and conservative categorical-label normalization are
  registered through the orchestration layer.

### Validation and output

```text
validation/
|-- __init__.py
`-- post_clean_validator.py

output/
|-- __init__.py
`-- clean_dataset_exporter.py
```

- `post_clean_validator.py` re-profiles the cleaned copy and checks dimensions,
  missingness, duplicate counts, dtypes, cardinality, unexpected changes, and
  raw-input preservation.
- `clean_dataset_exporter.py` writes a new `cleaned_dataset.csv` and the complete
  `cleaning_report.json`; it never intentionally overwrites the source dataset.

## Pipeline 1 decision and execution relationship

```text
DetectedIssue
  |
  +-- deterministic Tier 1 decision, when policy is conclusive
  |
  `-- structured Tier 2 recommendation, when context is required
          |
          +-- schema and issue-ID validation
          +-- issue-specific readiness alignment
          +-- deterministic risk/review policy
          `-- per-issue retry or preservation fallback
                      |
                      v
              CleaningDecision
                      |
          semantic + mechanical authorization
                      |
              CleaningInstruction
                      |
              allow-listed executor
                      |
               ExecutionResult
```

For Tier 2 median imputation specifically, execution requires all three of the
following: an explicit LLM recommendation using `strategy="median"`, positive
semantic eligibility, and a complete pass of the existing mechanical imputation
contract. False or unknown semantic eligibility preserves the data.

## `intelligence/`

### LLM abstraction

```text
intelligence/llm/
|-- __init__.py
|-- provider.py
|-- ollama_provider.py
`-- mock_provider.py
```

- `provider.py` defines the provider-independent structured-generation
  interface used by pipeline components.
- `ollama_provider.py` submits structured-output requests to local Ollama,
  validates responses against Pydantic schemas, and exposes native Ollama timing
  telemetry for diagnostics.
- `mock_provider.py` supplies deterministic test responses without a live model.

### Retrieval-augmented generation

```text
intelligence/rag/
|-- __init__.py
|-- knowledge_loader.py
|-- knowledge_chunker.py
|-- embedding_text_builder.py
|-- embeddings.py
|-- vector_store.py
|-- retriever.py
`-- mock_retriever.py
```

- `knowledge_loader.py` parses knowledge Markdown and validates its front matter.
- `knowledge_chunker.py` splits documents into retrievable sections/chunks.
- `embedding_text_builder.py` builds embedding input from knowledge metadata and
  content.
- `embeddings.py` wraps the configured embedding model.
- `vector_store.py` stores embeddings and performs similarity search in memory.
- `retriever.py` indexes the knowledge base and returns ranked, optionally
  category-filtered knowledge chunks.
- `mock_retriever.py` provides deterministic retrieval behavior for tests.

### Shared intelligence schemas

```text
intelligence/schemas/
`-- __init__.py
```

This package is reserved for model/provider-independent intelligence contracts.

## `knowledge_base/`

```text
knowledge_base/
|-- README.md
|-- MANIFEST.md
|-- KNOWLEDGE_SCHEMA.md
|-- semantics/
|   |-- core_types/
|   |-- concepts/
|   `-- structural_roles/
|-- data_quality/
|-- cleaning_methods/
|   |-- categorical/
|   |-- conversions/
|   |-- duplicates/
|   |-- identifiers/
|   |-- missing_values/
|   |-- numeric/
|   |-- outliers/
|   |-- temporal/
|   `-- text/
|-- decision_rules/
|-- domains/
`-- sources/
```

- `README.md` explains the purpose and organization of the knowledge base.
- `MANIFEST.md` inventories the knowledge collection.
- `KNOWLEDGE_SCHEMA.md` defines the Markdown/front-matter contract and
  `knowledge_id` conventions.
- `semantics/core_types/` defines broad semantic types such as identifier,
  categorical, currency, percentage, temporal, text, and measurement types.
- `semantics/concepts/` describes domain-neutral concepts such as entity
  references, classification codes, amounts, quantities, contact fields, names,
  locations, and ratings.
- `semantics/structural_roles/` distinguishes primary, foreign, and transaction
  identifiers from category, description, measurement, and timestamp roles.
- `data_quality/` describes issue families such as missingness, mixed formats,
  type mismatch, duplicates, outliers, invalid values, and representation
  inconsistencies.
- `cleaning_methods/` documents candidate transformations and their safeguards.
- `decision_rules/` provides preservation, evidence, uncertainty, information
  loss, recoverability, and semantic-constraint guidance.
- `domains/` provides cross-domain background for retrieval. Domain examples are
  supporting context, not deterministic mappings.
- `sources/README.md` is the source-document area guide.

Knowledge documents do not execute transformations. They are retrieved as
supporting definitions and context; observed dataset evidence remains
authoritative.

## `database/`

```text
database/
|-- __init__.py
|-- base.py
|-- connection.py
|-- create_tables.py
|-- models/
|   |-- __init__.py
|   |-- dataset.py
|   |-- dataset_version.py
|   |-- column.py
|   |-- dataset_profile.py
|   |-- column_profile.py
|   |-- pipeline_run.py
|   |-- cleaning_decision.py
|   `-- audit_event.py
`-- repositories/
    |-- __init__.py
    |-- dataset_repository.py
    `-- profile_repository.py
```

- `base.py` defines the shared SQLAlchemy declarative base.
- `connection.py` configures database engine/session creation.
- `create_tables.py` creates the declared database tables.
- `models/dataset.py` stores registered source-dataset metadata.
- `models/dataset_version.py` represents versioned dataset artifacts.
- `models/column.py` represents dataset columns.
- `models/dataset_profile.py` and `column_profile.py` persist profiling results.
- `models/pipeline_run.py` records pipeline execution state.
- `models/cleaning_decision.py` persists cleaning-decision lineage.
- `models/audit_event.py` records auditable system events.
- `dataset_repository.py` implements dataset registration, lookup, listing, and
  status updates.
- `profile_repository.py` implements profile persistence/retrieval operations.

The current file-based Pipeline 1 run can operate independently of this database
layer; database integration remains a separate concern.

## `benchmarks/`

```text
benchmarks/pipeline1_semantics/
|-- run_benchmark_v001.py
|-- ollama_prompt_cache_diagnostic.py
|-- ollama_prompt_cache_diagnostic.json
`-- benchmark_v001_*/
    |-- manifest.json
    |-- datasets.jsonl
    |-- columns.jsonl
    |-- errors.jsonl
    `-- summary.json
```

- `run_benchmark_v001.py` runs the versioned Architecture A semantic benchmark
  and records dataset/column outputs and timing.
- `ollama_prompt_cache_diagnostic.py` performs the real prompt-cache experiment
  using existing semantic prompts.
- `ollama_prompt_cache_diagnostic.json` contains its captured results.
- `benchmark_v001_*` directories are immutable or experimental benchmark output
  sets, including baseline, smoke, telemetry, timing, and compact-prompt runs.
  They are artifacts, not runtime modules.

## Pipeline 2 and Pipeline 3

```text
pipelines/pipeline2_prediction/
|-- __init__.py
|-- README.md
`-- models/
    `-- random_forest.py

pipelines/pipeline3_results/
|-- __init__.py
`-- README.md
```

- Pipeline 2 is reserved for prediction/modeling after cleaning;
  `random_forest.py` is the current initial model implementation.
- Pipeline 3 is reserved for presenting, evaluating, or exporting downstream
  prediction results.
- Neither pipeline currently has the same end-to-end maturity as Pipeline 1.

## `shared/`

```text
shared/
|-- __init__.py
|-- constants.py
|-- exceptions.py
|-- logging.py
`-- utils.py
```

- `constants.py` holds values shared across modules.
- `exceptions.py` defines shared exception types.
- `logging.py` centralizes logging setup.
- `utils.py` contains general-purpose helpers.

## `tests/`

```text
tests/
|-- datasets/
|   |-- valid/                   Supported CSV/XLSX fixtures across domains
|   `-- invalid/                 Unsupported or invalid-input fixtures
|-- unit/                        Focused deterministic component tests
|-- integration/                 End-to-end and live-provider tests/scripts
|-- pipeline1_multidomain/       13-domain evaluation runner and consolidator
|-- pipeline2/                   Prediction-stage tests
`-- __init__.py
```

### Unit tests

The unit suite covers:

- ingestion/profile behavior;
- semantic contracts, evidence, retrieval, and both semantic architectures;
- knowledge loading, chunking, embeddings, and vector search;
- deterministic issue detection;
- Tier 1/Tier 2 cleaning decisions and per-issue failure isolation;
- instruction compilation and execution;
- date parsing, numeric coercion, and median-imputation safeguards;
- Pipeline 1 validation/export behavior;
- Benchmark V001 behavior.

### Integration tests and diagnostic scripts

- `test_pipeline1_vertical_slice.py` verifies the complete mechanical Pipeline 1
  route and output artifacts.
- Semantic integration tests exercise dataset understanding, column
  understanding, role resolution, RAG indexing/retrieval, and Ollama.
- `run_parse_dates_live.py`, `run_coerce_dtype_live.py`, and
  `run_impute_missing_live.py` validate individual cleaning actions against live
  datasets.
- `run_missing_value_tier2_reasoning_live.py` runs the narrow nine-case Qwen
  diagnostic and records raw responses, independent validation, smaller retries,
  timeouts, recoveries, and per-issue fallbacks.
- Generated `cleaned_dataset.csv`, `cleaning_report.json`, and diagnostic JSON
  files under integration output directories are test artifacts.

### Multi-domain evaluation

```text
tests/pipeline1_multidomain/
|-- run_multidomain_test.py
|-- consolidate_results.py
`-- results*/
```

- `run_multidomain_test.py` runs Pipeline 1 across the 13 domain fixtures.
- `consolidate_results.py` combines per-dataset reports into JSONL and summary
  artifacts.
- `results*/` contains generated cleaned datasets, reports, checkpoints, issue
  records, decisions, executions, and validation results. These directories are
  evidence from runs, not production source code.

## Important architectural boundaries

1. The raw input is treated as immutable; cleaning operates on a copy.
2. Issue detection reports candidates and evidence but does not clean data.
3. Semantic inference describes meaning; it does not execute transformations.
4. RAG supplies knowledge, not observed facts or execution authority.
5. Tier 2 supplies contextual recommendations, not arbitrary code.
6. Pydantic contracts, readiness alignment, deterministic policy, instruction
   compilation, and action validators constrain recommendations.
7. Only action IDs in the registry can execute.
8. An unresolved or unsafe case is preserved, flagged, or returned as
   insufficient evidence.
9. Every successful run exports a new dataset artifact and an auditable report.

## Current maturity summary

| Area | Current state |
|---|---|
| Pipeline 1 ingestion/profiling | Implemented |
| Deterministic issue detection | Implemented |
| Architecture B batched semantics | Implemented and active |
| Architecture A semantics | Retained for benchmarks/regression |
| Hybrid cleaning decisions | Implemented |
| Per-issue Tier 2 retry isolation | Implemented |
| Allow-listed deterministic actions | Implemented for current MVP actions |
| Post-clean validation/export | Implemented |
| Database models/repositories | Scaffolded/partially integrated |
| HTTP API/UI | Not implemented |
| Pipeline 2 prediction | Early scaffold |
| Pipeline 3 results | Early scaffold |

