---
knowledge_id: DQ-014
category: data_quality
topic: impossible_relationships
version: 1.0
status: active
---

# Impossible Relationships

## Definition
Two or more fields jointly violate a deterministic or strongly supported semantic relationship.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Distinguish semantic meaning from physical storage type and from superficially similar value patterns.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Examples: start after end where impossible, min greater than max, impossible formula reconciliation.

## Diagnostic Evidence
Verify relationship applicability and tolerances.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Correct from authoritative fields if possible; otherwise flag affected values/record.

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
