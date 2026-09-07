---
knowledge_id: DOM-006
category: domain
topic: cybersecurity
version: 1.0
status: active
---

# Cybersecurity

## Definition
Security events, alerts, identities, hosts, network activity, vulnerabilities and responses.

## Semantic Indicators
Common concepts: Event IDs, timestamps, IPs, ports, users, hosts, severity, action, protocol, detection status.

## Important Distinctions
Possible row grain: Rows often represent events, alerts, flows, authentication attempts or findings. Domain indicators are supporting evidence, not keyword rules.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Missingness, invalid values, implausible values, sentinel values, inconsistent representations, typographical errors, parsing errors, mixed units, and cross-column contradictions may occur.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Event timestamps, source/destination fields, protocol-port relationships and alert lifecycle fields may relate.

## Cleaning Decision Guidance
Use domain knowledge to generate candidate semantic interpretations and constraints. Never override observed dataset evidence or assume all listed concepts are present.

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
