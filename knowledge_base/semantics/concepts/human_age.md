---
knowledge_id: CON-001
category: semantic_concept
topic: human_age
version: 1.0
status: active
---

# Human Age

## Definition
A person’s age at a relevant reference time, commonly measured in years.

## Semantic Indicators
Names such as age, patient_age, employee_age plus person-level dataset context.

## Important Distinctions
Age is a measurement, not an identifier. It may be stored as integer, decimal, text, range, or category.

## Quality Expectations
Non-negative and plausible for the represented population; exact plausible bounds depend on context.

## Possible Data Quality Problems
Implausible values such as 999, negative ages, sentinel values, unit confusion, stale age inconsistent with dates.

## Diagnostic Evidence
Check population, distribution, date_of_birth, record/event date, repeated sentinel patterns.

## Cross-Column Relationships
Age can often be checked against date_of_birth and a reference/event date.

## Cleaning Decision Guidance
If a value is implausible, first determine whether it is sentinel/error and whether authoritative dates allow reconstruction. Otherwise flag or convert to missing only if justified.

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
