---
knowledge_id: SEM-010
category: semantic
topic: percentage
version: 1.0
status: active
---

# Percentage

## Definition
A relative quantity expressed as a proportion, rate, or percent.

## Semantic Indicators
Names containing percent/rate/ratio/discount; % signs; values commonly on 0–1 or 0–100 scales.

## Important Distinctions
Distinguish proportions from counts and arbitrary numeric scores. Do not assume all percentages must be 0–100; domain context matters.

## Quality Expectations
Representation scale should be consistent and semantically plausible.

## Possible Data Quality Problems
Mixed 0–1 and 0–100 scales, embedded % signs, impossible rates, rounding inconsistencies.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Use related columns when they provide semantically meaningful validation or reconstruction evidence; do not invent relationships from proximity alone.

## Cleaning Decision Guidance
Normalize scale only when the intended convention can be inferred reliably.

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
