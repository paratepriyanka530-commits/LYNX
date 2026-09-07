---
knowledge_id: ROL-005
category: structural_role
topic: category
version: 1.0
status: active
---

# Category Role

## Definition
Groups rows into semantic classes or states.

## Semantic Indicators

A category column represents a discrete set of labels, groups, classes,
states, types, or named groups used to organize or segment records.

Common evidence includes:

- values repeat across multiple rows;
- the number of distinct values is relatively small compared with the
  number of records;
- values behave as labels rather than quantities or free-form text;
- examples may include departments, cities, regions, statuses, product
  types, customer segments, classes, or similar group labels;
- the column is commonly useful for grouping, filtering, aggregation,
  or comparison.

A string-valued column should not automatically be treated as a
description. Repeated discrete labels are stronger evidence for a
category role than for a description role.

Encoded categorical values may also use codes rather than human-readable
labels. A code does not necessarily function as an identifier.

When repeated codes represent classes, diagnoses, statuses, types, groups,
levels, or other finite classifications, the structural role is typically
category rather than primary_identifier.

Distinguish categorical codes from identifiers by asking what the value
identifies. If it classifies an attribute of the row rather than uniquely
identifying the row's main entity or record, category is the stronger role.
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
