---
knowledge_id: DOM-001
category: domain
topic: retail
version: 1.0
status: active
---

# Retail

## Definition
Sales, stores, products, customers, orders, line items, prices, discounts, payments and channels.

## Semantic Indicators
Common concepts: Order/transaction IDs, product/customer/store references, quantity, unit price, discounts, totals, dates, payment methods.

## Important Distinctions
Possible row grain: Rows may represent transactions, order lines, products, customers, stores, inventory events or aggregates. Domain indicators are supporting evidence, not keyword rules.

## Quality Expectations
Values should be internally coherent, semantically plausible, consistently represented, and compatible with relevant cross-column relationships.

## Possible Data Quality Problems
Missingness, invalid values, implausible values, sentinel values, inconsistent representations, typographical errors, parsing errors, mixed units, and cross-column contradictions may occur.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Quantity × unit price adjusted by discount/tax may relate to totals; product/store/customer IDs often reference entities.

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
