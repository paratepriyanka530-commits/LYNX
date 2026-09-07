---
knowledge_id: DQ-001
category: data_quality
topic: missing_values
version: 1.0
status: active
---

# Missing Values

## Definition
Absence of an observed value, including explicit nulls and semantically missing placeholders.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Distinguish true missingness from empty strings, sentinel values, not-applicable states and parsing failures.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Unnecessary imputation can invent information.

## Diagnostic Evidence
Check missing percentage, pattern by groups/rows, semantic meaning, downstream need, and whether another column can reconstruct the value.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Impute only when justified; otherwise preserve, flag, or remove rows/columns according to evidence and purpose.

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
