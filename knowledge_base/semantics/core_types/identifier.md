---
knowledge_id: SEM-001
category: semantic
topic: identifier
version: 1.0
status: active
---

# Identifier

## Definition
A value whose main purpose is to distinguish, reference, or link one particular
entity, event, record, transaction, or object instance. Arithmetic magnitude
usually has no domain meaning.

## Semantic Indicators
The decisive evidence is what the value denotes: an identifier points to a
particular instance that can in principle be distinguished from other instances.
High uniqueness may support row identity. Repeated values may support a reference
to the same related entity across several rows.

Names such as id, key, code, number, or reference and identifier-shaped values
are supporting evidence only. Numeric, alphabetic, alphanumeric, dotted,
prefixed, or abbreviated representation does not establish identifier meaning.

## Important Distinctions
Distinguish an instance reference from membership in a reusable class or code
system. A value is not an identifier merely because it looks like a code or its
column name contains id, code, key, number, or reference.

If the same value denotes a class, type, status, state, level, taxonomy member,
or other category that can apply to many records, its semantic type is normally
categorical rather than identifier. If it denotes one particular customer,
account, device, employee, event, order, or other instance, identifier is the
stronger interpretation.

Also distinguish identifiers from measurements and ordinal numbers. Numeric
storage does not make an identifier a measurement.

## Quality Expectations
Preserve exact identity, leading zeros, case when meaningful, and referential consistency.

## Possible Data Quality Problems
Duplicate identifiers, missing identifiers, formatting drift, accidental numeric conversion, broken references.

## Diagnostic Evidence
Check uniqueness, repetition, prefixes, paired entity tables, and use across related columns.

## Cross-Column Relationships
Primary identifiers tend to identify row entities; foreign identifiers reference another entity; transaction identifiers group transaction-related rows.

## Cleaning Decision Guidance
Standardize only when identity equivalence is supported. Never impute identifiers using mean/median or fabricate new source identities.

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
