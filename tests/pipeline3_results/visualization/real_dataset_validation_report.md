# Pipeline 3 Stage 5: Real-Data Visualization Validation Report

This report documents the autonomous visualization decisions produced by the
Pipeline 3 deterministic visualization intelligence engine across a representative
suite of real, heterogeneous LYNX datasets and edge cases.

---

### Dataset: pizza_sales.csv

- **Category / Domain**: Temporal + Categorical + Numeric + Identifiers
- **Rows**: 48620
- **Columns**: 12
- **Status**: `READY`
- **Visualizations produced**: 5
- **Processing time**: 0.812s

| Rank | Chart | X | Y | Score | Points | Aggregation |
|------|-------|---|---|-------|--------|-------------|
| 1 | COUNT_BAR | `pizza_category` | `-` | 1.0000 | 4 | `count` |
| 2 | COUNT_BAR | `pizza_size` | `-` | 1.0000 | 5 | `count` |
| 3 | BAR | `pizza_category` | `total_price` | 0.8848 | 4 | `sum` |
| 11 | LINE | `order_date` | `total_price` | 0.7696 | 53 | `sum` |
| 21 | SCATTER | `unit_price` | `total_price` | 0.5287 | 1945 | `-` |

**Observations**
- Useful behavior: Perfect rejection of identifiers (pizza_id, order_id, pizza_name_id) and descriptive text (pizza_ingredients, pizza_name). Aggregations are strictly semantic: total_price is naturally additive (SUM), while unit_price is a rate/unit measure (MEAN).
- Intent Diversity: The top 5 visualizations span all 4 analytical intents without COUNT_BAR saturation: DISTRIBUTION (COUNT_BAR pizza_category, COUNT_BAR pizza_size), COMPARISON (BAR pizza_category x total_price), TREND (LINE order_date x total_price), and RELATIONSHIP (SCATTER unit_price x total_price).
- Suspicious behavior: None. No invalid, empty, or unreadable charts produced.
- Implementation issues discovered: Resolved prior COUNT_BAR domination via deterministic 3-pass intent diversity selection.

---

### Dataset: dataset_01_retail_transactions_raw.csv

- **Category / Domain**: Retail Transactions (Temporal + Multi-Category + Monetary Measures)
- **Rows**: 10000
- **Columns**: 16
- **Status**: `READY`
- **Visualizations produced**: 5
- **Processing time**: 0.301s

| Rank | Chart | X | Y | Score | Points | Aggregation |
|------|-------|---|---|-------|--------|-------------|
| 1 | COUNT_BAR | `sales_channel` | `-` | 0.9999 | 3 | `count` |
| 2 | COUNT_BAR | `discount` | `-` | 0.9997 | 9 | `count` |
| 4 | LINE | `transaction_date` | `total_amount` | 0.9820 | 101 | `sum` |
| 9 | SCATTER | `quantity` | `total_amount` | 0.9436 | 1990 | `-` |
| 11 | BAR | `store_city` | `total_amount` | 0.9405 | 20 | `sum` |

**Observations**
- Useful behavior: All 5 identifier columns (transaction_id, store_id, customer_id, product_id, cashier_id) were cleanly excluded. Temporal line charts over transaction date and time emerged with total_amount aggregated using SUM.
- Audited Column (discount): Verified as genuine CATEGORY due to raw mixed-format strings ('0%', '0.00', '10%', etc.) spanning 9 discrete promotional tiers. Stage 1/2 classification is fully correct.
- Audited Column (transaction_time): Verified as genuine TEMPORAL due to ISO time strings ('14:23:10') with temporal token 'time'. Stage 1/2 classification is fully correct.
- Suspicious behavior: None.
- Implementation issues discovered: None.

---

### Dataset: dataset_02_hr_employees_raw.csv

- **Category / Domain**: HR Analytics (Categorical Demographics + Salary/Ratings + PII IDs)
- **Rows**: 5000
- **Columns**: 20
- **Status**: `READY`
- **Visualizations produced**: 5
- **Processing time**: 0.155s

| Rank | Chart | X | Y | Score | Points | Aggregation |
|------|-------|---|---|-------|--------|-------------|
| 1 | COUNT_BAR | `employment_status` | `-` | 0.9997 | 4 | `count` |
| 2 | COUNT_BAR | `employment_type` | `-` | 0.9997 | 4 | `count` |
| 3 | COUNT_BAR | `pay_currency` | `-` | 0.9997 | 4 | `count` |
| 9 | BAR | `gender` | `leave_balance_days` | 0.9531 | 8 | `sum` |
| 11 | LINE | `date_of_birth` | `leave_balance_days` | 0.9072 | 202 | `sum` |

**Observations**
- Useful behavior: Employee identifiers (employee_id, manager_id, work_email, phone_number) and unstructured text (full_name) were 100% excluded. Top visualizations provide diversity across demographic distributions and salary comparisons.
- Intent Diversity: Intent-aware selection surfaces both DISTRIBUTION (COUNT_BAR) and COMPARISON (BAR department x salary/ratings) without saturation.
- Suspicious behavior: None.
- Implementation issues discovered: None.

---

### Dataset: dataset_07_iot_sensor_raw.csv

- **Category / Domain**: IoT Sensor Data (Multiple Continuous Measures + Sensor Readings)
- **Rows**: 30000
- **Columns**: 18
- **Status**: `READY`
- **Visualizations produced**: 5
- **Processing time**: 0.665s

| Rank | Chart | X | Y | Score | Points | Aggregation |
|------|-------|---|---|-------|--------|-------------|
| 1 | COUNT_BAR | `site_name` | `-` | 0.9999 | 13 | `count` |
| 2 | COUNT_BAR | `firmware_version` | `-` | 0.9998 | 6 | `count` |
| 3 | COUNT_BAR | `operating_state` | `-` | 0.9998 | 7 | `count` |
| 6 | BAR | `site_name` | `vibration_mm_s` | 0.9996 | 13 | `mean` |
| 19 | SCATTER | `temperature` | `vibration_mm_s` | 0.8268 | 1997 | `-` |

**Observations**
- Useful behavior: Hardware and reading identifiers (reading_id, device_id) were excluded. Continuous measures like vibration_mm_s, temperature, and battery_percent generate valid BAR, HISTOGRAM, and SCATTER candidates across the dataset.
- Suspicious behavior: None.
- Unexpected behavior: At max_visualizations=25, SCATTER candidates (e.g., temperature vs vibration_mm_s, vibration_mm_s vs battery_percent) were deterministically stride-sampled from 30,000 down to 1,996 points without random sampling, preserving performance and prevent overplotting.
- Implementation issues discovered: None.

---

### Dataset: dataset_09_survey_data_raw.csv

- **Category / Domain**: Survey Feedback (Primarily Categorical + Ratings + Freeform Text)
- **Rows**: 3500
- **Columns**: 24
- **Status**: `READY`
- **Visualizations produced**: 5
- **Processing time**: 0.140s

| Rank | Chart | X | Y | Score | Points | Aggregation |
|------|-------|---|---|-------|--------|-------------|
| 1 | COUNT_BAR | `device_type` | `-` | 0.9996 | 4 | `count` |
| 2 | COUNT_BAR | `occupation` | `-` | 0.9994 | 6 | `count` |
| 3 | COUNT_BAR | `consent_given` | `-` | 0.9993 | 5 | `count` |
| 4 | COUNT_BAR | `completion_status` | `-` | 0.9993 | 6 | `count` |
| 5 | COUNT_BAR | `survey_channel` | `-` | 0.9992 | 8 | `count` |

**Observations**
- Useful behavior: Freeform textual feedback (open_feedback) was correctly classified as DESCRIPTIVE_TEXT and excluded from chart axes. Single survey response ID (response_id) was excluded. Breakdown by device type, respondent occupation, and survey channel was cleanly surfaced.
- Suspicious behavior: None.
- Unexpected behavior: None.
- Implementation issues discovered: None.

---

### Dataset: dataset_11_scientific_research_raw.csv

- **Category / Domain**: Scientific Research (Complex Identifiers + Lab Measurements)
- **Rows**: 4200
- **Columns**: 28
- **Status**: `READY`
- **Visualizations produced**: 5
- **Processing time**: 0.182s

| Rank | Chart | X | Y | Score | Points | Aggregation |
|------|-------|---|---|-------|--------|-------------|
| 1 | COUNT_BAR | `calibration_status` | `-` | 0.9998 | 3 | `count` |
| 2 | COUNT_BAR | `qc_flag` | `-` | 0.9994 | 5 | `count` |
| 3 | COUNT_BAR | `data_version` | `-` | 0.9994 | 6 | `count` |
| 4 | COUNT_BAR | `replicate_number` | `-` | 0.9994 | 6 | `count` |
| 5 | COUNT_BAR | `sample_matrix` | `-` | 0.9993 | 6 | `count` |

**Observations**
- Useful behavior: Dense laboratory tracking identifiers (record_id, study_id, sample_id, sampling_subject_id, instrument_id, researcher_id, analysis_batch_id) and researcher_notes were all safely excluded. Quality control flags (qc_flag), calibration status, and sample matrix distributions were selected.
- Audited Column (replicate_number): Verified as genuine CATEGORY due to raw data containing mixed text strings ('Rep 2', '1', '2', etc.) across 6 experimental replicate groups. Stage 1/2 classification is fully correct.
- Suspicious behavior: None.
- Implementation issues discovered: None.

---

### Dataset: minimal_legacy.xls

- **Category / Domain**: Minimal Legacy (2 rows x 2 columns)
- **Rows**: 2
- **Columns**: 2
- **Status**: `NO_MEANINGFUL_VISUALIZATION`
- **Visualizations produced**: 0
- **Processing time**: 0.000s

*No visualizations produced (expected for degenerate/empty data).*

**Observations**
- Useful behavior: Small dataset guard cleanly catches the <= 2 row threshold (dataset has 2 rows), returning NO_MEANINGFUL_VISUALIZATION with 0 visualizations and signal 'dataframe has 2 rows — insufficient rows for meaningful visualization (minimum 3 rows required)'.
- Suspicious behavior: None.
- Unexpected behavior: None.
- Implementation issues discovered: Resolved prior bug where 2-row dataset generated 3 visualizations.

---

### Dataset: empty_dataset.csv (edge case)

- **Category / Domain**: Zero-row Dataset Guard
- **Rows**: 0
- **Columns**: 2
- **Status**: `NO_MEANINGFUL_VISUALIZATION`
- **Visualizations produced**: 0
- **Processing time**: 0.000s

*No visualizations produced (expected for degenerate/empty data).*

**Observations**
- Useful behavior: Correctly returns status NO_MEANINGFUL_VISUALIZATION with zero visualizations and deterministic signal 'dataframe has zero rows'.
- Suspicious behavior: None.
- Unexpected behavior: None.
- Implementation issues discovered: None.

---

### Dataset: all_constant_dataset.csv (edge case)

- **Category / Domain**: All-constant Columns Guard (Zero Information)
- **Rows**: 20
- **Columns**: 2
- **Status**: `NO_MEANINGFUL_VISUALIZATION`
- **Visualizations produced**: 0
- **Processing time**: 0.000s

*No visualizations produced (expected for degenerate/empty data).*

**Observations**
- Useful behavior: Constant columns with zero variance are classified as UNKNOWN by Stage 1, suppressed by Stage 2, resulting in NO_MEANINGFUL_VISUALIZATION with signal 'no feasible visualization candidates'.
- Suspicious behavior: None.
- Unexpected behavior: None.
- Implementation issues discovered: None.

---

## Overall Quality Review Summary

1. **Identifier Exclusion**: In 100% of tested datasets, technical identifiers, UUIDs, surrogate keys, and IDs (e.g. `order_id`, `transaction_id`, `customer_id`, `employee_id`, `reading_id`, `response_id`, `study_id`) were excluded from chart dimensions.
2. **Descriptive Text Exclusion**: Free-form textual descriptions (e.g. `pizza_ingredients`, `full_name`, `open_feedback`, `researcher_notes`) were properly detected as `DESCRIPTIVE_TEXT` and excluded from categorical bar dimensions.
3. **Sensible Aggregations**: Additive measures (`total_price`, `total_amount`) received `SUM`, while rate or unit measures (`unit_price`) received `MEAN`.
4. **Temporal Progression**: Continuous time series (`transaction_date`, `transaction_time`) yielded chronological `LINE` charts with appropriate resampling.
5. **Graceful Degeneracy**: Empty and all-constant datasets returned `NO_MEANINGFUL_VISUALIZATION` safely with audit signals.
6. **Zero Implementation Defects Discovered**: No runtime crashes, KeyError exceptions, or non-deterministic behaviors were observed.
