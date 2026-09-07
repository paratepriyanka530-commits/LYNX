---
knowledge_id: CLN-MIS-001
category: cleaning_method
topic: mean_imputation
version: 1.0
status: active
---

# Mean Imputation

## Definition
Replace missing numeric observations with the arithmetic mean of valid observations.

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
Use only for appropriate numeric measurements when missingness is limited, distribution is reasonably suitable, no stronger reconstruction exists, and the objective permits imputation.

## When NOT to Clean
Avoid for identifiers, skewed/heavy-tailed values, meaningful missingness, group-dependent distributions, or high missingness. Validate distributional distortion.

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
