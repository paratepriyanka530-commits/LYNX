---
knowledge_id: CON-009
category: semantic_concept
topic: classification_code
version: 1.0
status: active
---

# Classification Code

## Definition
A classification code is an encoded member of a reusable conceptual vocabulary.
It assigns a record or attribute to a class, type, status, state, level, taxonomy,
or category rather than designating one particular entity instance.

## Semantic Indicators
Evidence for a classification code includes:

- the same code applies to many independent records;
- values come from a bounded or governed vocabulary, even when not all possible
  members appear in the observed sample;
- the code supports grouping, filtering, comparison, or interpretation of rows;
- values describe what kind of record, condition, event, product, risk, defect,
  activity, or state is represented;
- a paired label or description may expand the code's class meaning;
- uniqueness is not the purpose of the field.

Classification codes can be numeric, alphabetic, alphanumeric, dotted,
hierarchical, prefixed, or abbreviated. Their encoded appearance does not imply
identifier semantics.

## Important Distinctions
Ask what one value denotes.

If it denotes one particular entity, object, record, event, or transaction that
could be referenced again, it is an entity identifier or reference. If it denotes
a reusable class that multiple instances can share, it is a classification code.

Repetition supports classification but is not decisive because entity references
can also repeat. Conversely, high cardinality does not rule out classification
when the vocabulary is a large taxonomy.

## Cross-Domain Illustrations
Examples of classification concepts include risk classes and transaction types
in finance; event types, severity codes, and attack categories in cybersecurity;
product classes, channel codes, and store types in retail; employment statuses
and job families in HR; defect types, machine states, and material classes in
manufacturing; and sample classes, experimental conditions, and scientific
taxonomies. A diagnostic classification is another illustration of the same
general concept.

These examples are not name-based rules. Dataset evidence must establish whether
the values denote shared classes rather than particular instances.

## Expected Semantic Interpretation
When supported by the evidence, use semantic type categorical and value function
classification. Structural role is determined separately.
