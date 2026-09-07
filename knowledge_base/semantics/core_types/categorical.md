---
knowledge_id: SEM-002
category: semantic
topic: categorical
version: 1.0
status: active
---

# Categorical

## Definition
A variable representing membership in a finite or conceptually discrete set of
classes, labels, states, types, levels, taxonomies, or groups. Category values
may be human-readable labels or compact encoded values.

## Semantic Indicators
Common evidence includes repeated values, a bounded or conceptually discrete
vocabulary, and values used to group or describe records rather than designate
particular entity instances. The same category value can legitimately apply to
many rows.

Encoded categories may be numeric, alphabetic, alphanumeric, dotted,
hierarchical, prefixed, or abbreviated. Status, state, type, class, severity,
risk level, industry, occupation, region classification, experimental condition,
and taxonomy fields are possible categorical concepts when supported by the
dataset evidence.

Low or moderate cardinality is useful evidence but is not required: a controlled
taxonomy can contain many possible members, and an entity reference can repeat.

## Important Distinctions
Distinguish a reusable classification from an entity identifier. A categorical
code denotes membership in a conceptual class, state, type, or taxonomy; an
identifier denotes one particular entity, record, event, transaction, or object.

Code-like syntax and words such as code, id, key, or number do not decide between
these meanings. Ask whether the value designates an instance or classifies the
row or one of its attributes.

Also distinguish category from free text, boolean, ordinal rating, and numeric
measurement.

## Quality Expectations
Equivalent categories should be represented consistently while preserving meaningful distinctions.

## Possible Data Quality Problems
Case variants, whitespace, spelling variants, synonyms, obsolete labels, unexpected categories.

## Diagnostic Evidence
Use frequency, normalized forms, edit similarity, domain vocabulary, and related fields.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Merge/standardize categories only when equivalence is supported; preserve genuinely distinct rare categories.

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
