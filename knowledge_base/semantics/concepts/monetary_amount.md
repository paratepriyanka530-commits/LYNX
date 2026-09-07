---
knowledge_id: CON-002
category: semantic_concept
topic: monetary_amount
version: 1.0
status: active
---

# Monetary Amount

## Definition
A specific amount of money such as price, cost, salary, revenue, balance or charge.

## Semantic Indicators
Use column names, values, physical representation, uniqueness, neighboring columns, dataset context, and domain context as supporting evidence. No single indicator is proof.

## Important Distinctions
Distinguish unit price, total amount, balance, income and currency code.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Mixed currencies, misplaced decimals, signs, formatted strings, impossible formula relationships.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Quantity × unit_price may validate totals; tax/discount fields may affect formulas.

## Cleaning Decision Guidance
Prefer deterministic recomputation when an authoritative formula and inputs are known.

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
