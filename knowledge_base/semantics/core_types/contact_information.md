---
knowledge_id: SEM-012
category: semantic
topic: contact_information
version: 1.0
status: active
---

# Contact Information

## Definition
Information used to contact a person or organization, including email addresses, phone numbers and communication handles.

## Semantic Indicators
Email-like patterns, phone-like patterns, contact-prefixed names.

## Important Distinctions
Distinguish contact information from identifiers: a phone/email may uniquely identify a record but its semantic purpose is contact.

## Quality Expectations
Preserve meaningful symbols/country codes and validate structure without assuming existence or ownership.

## Possible Data Quality Problems
Malformed emails, inconsistent phone formatting, missing country codes, placeholder contacts.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Normalize formatting conservatively; never invent missing contact data.

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
