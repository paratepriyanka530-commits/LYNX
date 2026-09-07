---
knowledge_id: ROL-001
category: structural_role
topic: primary_identifier
version: 1.0
status: active
---

# Primary Identifier

## Definition
Identifies the principal row-level entity or record within the dataset.

## Semantic Indicators

A primary identifier uniquely identifies the main entity or record represented
by each row of a dataset.

Common evidence includes:

- values are unique or nearly unique across rows;
- the column name may contain identifier-like terms such as id, code, number,
  or key, but naming alone is not evidence that the column is the primary
  identifier;

- the identifier must correspond to the dataset's main row entity or record
  grain to qualify as a primary identifier;

- codes that identify attributes, classifications, categories, diagnoses,
  products, departments, locations, or other referenced concepts should not
  be treated as primary identifiers merely because they contain "code" or
  "id";
- values function as labels or references rather than quantities;
- arithmetic operations on the values would not be semantically meaningful;
- the identifier corresponds to the dataset's main row entity;
- repeated values may occur only when the dataset grain represents multiple
  records for the same entity, in which case another identifier role may be
  more appropriate.

A numeric physical dtype does not make an identifier a numeric measurement.
Identifier values should be interpreted according to their referential role,
not their numeric representation.

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
