---
knowledge_id: CLN-MIS-004
category: cleaning_method
topic: groupwise_imputation
version: 1.0
status: active
---

# Groupwise Imputation

## Definition
Impute using statistics conditioned on a semantically meaningful group.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Distinguish semantic meaning from physical storage type and from superficially similar value patterns.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Missingness, invalid values, implausible values, sentinel values, inconsistent representations, typographical errors, parsing errors, mixed units, and cross-column contradictions may occur.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Use when distributions differ materially by justified groups and group labels are reliable.

## When NOT to Clean
Do not create arbitrary groups solely to improve imputation.

## Evidence Required Before Modification
Confirm semantic type/concept, issue type, affected values, and method prerequisites before execution. Validate after execution.

## Reasoning Principles
- Physical dtype is not semantic meaning.
- Statistical outlier does not automatically mean error.
- Missing does not automatically mean impute.
- Never invent a replacement value.
- Prefer deterministic validation when possible.
- Preserve uncertainty and information.
- no_action is valid.
