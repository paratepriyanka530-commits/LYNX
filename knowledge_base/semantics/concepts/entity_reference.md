---
knowledge_id: CON-010
category: semantic_concept
topic: entity_reference
version: 1.0
status: active
---

# Entity Reference

## Definition
An entity reference denotes one particular identifiable entity, object, record,
event, or transaction instance. It preserves identity across observations or
links a row to the specific instance that participates in it.

## Semantic Indicators
Evidence for an entity reference includes:

- each value designates an individual instance rather than a kind or class;
- the same value may recur because the same entity participates in multiple
  records;
- values may support linkage between records or datasets;
- uniqueness may indicate the row-level instance, while repetition may indicate
  a related participating instance;
- arithmetic on the value has no semantic meaning even if it is stored as a
  number.

Customer, account, device, supplier, employee, patient, doctor, order, and event
identifiers illustrate possible referenced entities across domains. These are
illustrations of instance identity, not deterministic column-name mappings.

## Important Distinctions
An entity reference answers "which particular instance?" A classification code
answers "which reusable kind, class, state, type, level, or taxonomy member?"

Both can be repeated, encoded, alphanumeric, or called a code. Syntax and
cardinality alone cannot decide between them; the denoted concept is decisive.

Whether an entity reference identifies the dataset row entity or a different
participating entity is a structural-role question resolved later using dataset
row grain.

## Expected Semantic Interpretation
When supported by the evidence, use semantic type identifier and value function
entity_reference. Primary versus foreign identifier is not decided at the
semantic-concept stage.
