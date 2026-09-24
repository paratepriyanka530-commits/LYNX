# Pipeline 3: Results

Pipeline 3 provides autonomous results intelligence for LYNX, including Visualization Intelligence and explanation generation.

## Stage 1: Deterministic Column Intelligence

Stage 1 is the deterministic foundation that classifies tabular dataset dimensions into exactly seven analytical roles:

1. **`MEASURE`**: Quantitative/numerical values representing magnitudes, metrics, continuous measurements, or counts (e.g., `price`, `sales`, `quantity`, `revenue`).
2. **`CATEGORY`**: Discrete grouping attributes with bounded cardinality and repeated observations (e.g., `region`, `status`, `department`).
3. **`TEMPORAL`**: Date, time, timestamp, or temporal sequence attributes (e.g., `order_date`, `timestamp`, `year`).
4. **`IDENTIFIER`**: Primary keys, surrogate keys, UUIDs, foreign keys, or unique tracking tokens (e.g., `customer_id`, `order_id`, UUID strings).
5. **`DESCRIPTIVE_TEXT`**: Free-form text, multi-word notes, comments, descriptions, or reviews.
6. **`BOOLEAN`**: Binary boolean indicators (physical booleans, boolean strings like `{"true", "false"}`, or numeric `{0, 1}` supported by boolean names/semantics).
7. **`UNKNOWN`**: Columns where evidence is contradictory, all missing, zero-variance, or insufficient.

### Architecture & Design Rules

- **Zero API Coupling**: Pipeline 3 logic is self-contained under `pipelines/pipeline3_results/` and completely isolated from the API routing layer (`api/`) and frontend (`frontend/`).
- **Deterministic & Auditable**: Every classification decision produces explicit signal tokens (e.g. `["numeric_binary_values", "boolean_name_signal"]`) without invented confidence percentages or non-deterministic models.
- **Conservative Numeric {0, 1} Handling**: Numeric `{0, 1}` columns are classified as `BOOLEAN` only when boolean semantics are supported by column naming or Pipeline 1 evidence. Otherwise, they are classified conservatively as `CATEGORY`.
- **Conservative Temporal Detection**: Date parseability alone does not make an arbitrary string `TEMPORAL`. Classification requires a temporal naming signal or Pipeline 1 temporal evidence corroborated by strong parseability.
- **Graceful Pipeline 1 Evidence Reuse**: Accepts full, partial, or absent Pipeline 1 cleaning reports safely.
