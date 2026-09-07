---
knowledge_id: CON-006
category: semantic_concept
topic: email_address
version: 1.0
status: active
---

# Email Address

## Definition
A structured electronic mail contact address.

## Semantic Indicators
Values typically contain a local part, @ separator and domain; column name may indicate email.

## Important Distinctions
Semantic purpose is contact information even when values are unique.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Malformed structure, whitespace, case inconsistencies, placeholders, multiple emails in one cell.

## Diagnostic Evidence
Use deterministic syntax checks; syntax validity does not prove mailbox existence.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Trim accidental whitespace; preserve local-part semantics; do not fabricate corrections to ambiguous addresses.

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
