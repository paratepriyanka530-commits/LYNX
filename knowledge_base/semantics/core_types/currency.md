---
knowledge_id: SEM-009
category: semantic
topic: currency
version: 1.0
status: active
---

# Currency

## Definition
A monetary quantity whose magnitude represents money, often with an explicit or implicit currency unit.

## Semantic Indicators
Price, cost, revenue, salary, balance, amount; currency symbols/codes.

## Important Distinctions
Distinguish currency from generic numeric measurement and currency code categories.

## Quality Expectations
Currency unit, sign, decimal scale, and accounting conventions should be coherent.

## Possible Data Quality Problems
Mixed currencies, symbols embedded in numeric values, decimal/thousands separator ambiguity, unit-scale mismatch.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Currency code or country may clarify units; totals may relate to quantity and unit price.

## Cleaning Decision Guidance
Convert currencies only with an explicit policy/rate; strip formatting only after preserving currency semantics.

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
