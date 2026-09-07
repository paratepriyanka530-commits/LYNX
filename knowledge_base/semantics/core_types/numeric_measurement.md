---
knowledge_id: SEM-003
category: semantic
topic: numeric_measurement
version: 1.0
status: active
---

# Numeric Measurement

## Definition
A quantitative value whose magnitude and arithmetic relationships carry semantic meaning.

## Semantic Indicators
Numeric values; names indicating count, amount, duration, distance, score, temperature, rate, weight, etc.

## Important Distinctions
Distinguish measurements from numeric identifiers, categorical codes, ranks, currency and percentage semantics.

## Quality Expectations
Values should use coherent units, representation, precision, and plausible domain ranges.

## Possible Data Quality Problems
Outliers, impossible values, mixed units, decimal errors, sign errors, sentinel numbers, numeric strings.

## Diagnostic Evidence
Check distribution, units, neighboring fields, formulas, domain constraints, and repeated suspicious values.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Do not cap/remove extremes merely because they are statistical outliers. Correct only with semantic or relational evidence.

## When NOT to Clean
Do not modify unusual but plausible values, meaningful missingness, rare categories, valid alternate representations, or values whose intended correction cannot be supported.

## Evidence Required Before Modification
Destructive or value-changing actions require strong evidence. Weak evidence should lead to no_action, flag_for_review, or needs_additional_context.

## Reasoning Principles
- Physical dtype is not semantic meaning.
- Statistical outlier does not automatically mean error.
- Missing does not automatically mean impute.
- Never invent a replacement value.
- Prefer deterministic validation when possible.
- Preserve uncertainty and information.
- no_action is valid.
