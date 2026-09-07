---
knowledge_id: ROL-002
category: structural_role
topic: foreign_identifier
version: 1.0
status: active
---

# Foreign Identifier

## Definition
References another entity or table rather than defining the current row entity.

## Semantic Indicators

A foreign identifier identifies an entity related to the row's main entity
or record grain, rather than identifying the row itself.

Common evidence includes:

- the values reference another entity such as a patient, customer, employee,
  doctor, hospital, product, account, supplier, or similar related entity;
- the same identifier may legitimately appear in multiple rows because one
  related entity can participate in several records, events, transactions,
  encounters, or observations;
- the column is identifier-like, but its meaning does not correspond to the
  dataset's main row entity;
- uniqueness is therefore not required and repeated values can be expected;
- the column may conceptually link the dataset to another entity or table.

Distinguish a foreign identifier from a primary identifier by comparing the
entity identified by the column with the dataset's row entity.

If the dataset represents events, encounters, transactions, or observations,
an identifier referring to a participating entity should generally be
considered a foreign identifier rather than the primary identifier of the row.

## Important Distinctions
Assign roles only after considering row entity and dataset structure. A semantic type does not uniquely determine a role.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Missingness, invalid values, implausible values, sentinel values, inconsistent representations, typographical errors, parsing errors, mixed units, and cross-column contradictions may occur.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Role assignment should use established semantic type plus dataset grain/entity relationships. Do not label unrelated fields as transaction identifiers merely because a transaction exists somewhere in the dataset.

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
