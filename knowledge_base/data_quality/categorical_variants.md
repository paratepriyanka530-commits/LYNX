---
knowledge_id: DQ-007
category: data_quality
topic: categorical_variants
version: 1.0
status: active
---

# Categorical Variants

## Definition
Multiple labels that may represent the same category.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Distinguish semantic meaning from physical storage type and from superficially similar value patterns.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Case/spacing variants, abbreviations, synonyms, legacy labels.

## Diagnostic Evidence
Compare normalized forms, frequency, domain vocabulary and downstream distinctions.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Merge only proven equivalents.

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
