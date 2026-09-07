---
knowledge_id: DOM-003
category: domain
topic: healthcare
version: 1.0
status: active
---

# Healthcare

## Definition
Patients, encounters, diagnoses, procedures, providers, observations, medications and billing.

## Semantic Indicators
Common concepts: Patient/encounter/provider IDs, admission/discharge dates, diagnosis/procedure fields, clinical measurements, costs.

## Important Distinctions
Possible row grain: Rows may represent patients, encounters, procedures, claims, observations, prescriptions or billing events. Domain indicators are supporting evidence, not keyword rules.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Missingness, invalid values, implausible values, sentinel values, inconsistent representations, typographical errors, parsing errors, mixed units, and cross-column contradictions may occur.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Encounter dates, patient age/DOB, admission/discharge ordering, procedure/encounter references and billing totals may relate.

## Cleaning Decision Guidance
Use domain knowledge to generate candidate semantic interpretations and constraints. Never override observed dataset evidence or assume all listed concepts are present.

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
