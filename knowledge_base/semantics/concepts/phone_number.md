---
knowledge_id: CON-007
category: semantic_concept
topic: phone_number
version: 1.0
status: active
---

# Phone Number

## Definition
A structured telephone contact value, possibly including country/area codes and extensions.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Phone numbers are contact information, not numeric measurements; leading zeros and + signs matter.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Lost leading zeros, mixed country codes, punctuation variants, invalid lengths, placeholders.

## Diagnostic Evidence
Use region/domain context only as supporting evidence; avoid assuming a country without support.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Normalize formatting only after preserving country-code semantics.

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
