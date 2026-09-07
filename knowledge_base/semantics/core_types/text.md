---
knowledge_id: SEM-004
category: semantic
topic: text
version: 1.0
status: active
---

# Text

## Definition
Free-form or descriptive natural-language content whose meaning is primarily linguistic rather than membership in a small controlled set.

## Semantic Indicators
Sentences, descriptions, notes, comments, addresses, narratives, multiword variable content.

## Important Distinctions
Distinguish free text from categorical labels, structured codes, names, contact information, and delimited multi-label fields.

## Quality Expectations
Preserve linguistic content; normalize only representation defects that do not alter meaning.

## Possible Data Quality Problems
Encoding issues, whitespace, control characters, accidental duplication, corrupted characters, inconsistent casing where casing is not meaningful.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Avoid aggressive spelling correction or category merging in free text without strong evidence.

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
