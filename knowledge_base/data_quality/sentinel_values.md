---
knowledge_id: DQ-004
category: data_quality
topic: sentinel_values
version: 1.0
status: active
---

# Sentinel Values

## Definition
Special ordinary-looking values used to encode missing, unknown, not applicable, error or other states.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Sentinels can be numbers or strings and may overlap valid values.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Examples may include -999, 999, 0, N/A, unknown, ?, but none is universally a sentinel.

## Diagnostic Evidence
Look for repeated unusual values, documentation, impossible semantic fit, co-occurrence patterns and domain conventions.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Convert to explicit missing/state only when sentinel interpretation is well supported.

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
