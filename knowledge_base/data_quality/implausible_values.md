---
knowledge_id: DQ-003
category: data_quality
topic: implausible_values
version: 1.0
status: active
---

# Implausible Values

## Definition
Values that conflict strongly with inferred semantic/domain expectations but are not necessarily logically impossible.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Implausible differs from statistical outlier and proven invalid.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Age=999 may be implausible, sentinel, unit error, typo, or rare valid value in another concept.

## Diagnostic Evidence
Check semantic concept, population, distribution, sentinel patterns, units and cross-column evidence.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Do not guess. Reconstruct from authoritative evidence when possible; otherwise flag or treat as invalid/missing only when justified.

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
