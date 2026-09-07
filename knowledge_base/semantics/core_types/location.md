---
knowledge_id: SEM-011
category: semantic
topic: location
version: 1.0
status: active
---

# Location

## Definition
A geographic place, region, coordinate, address component, or spatial designation.

## Semantic Indicators
City/state/country/region/postal/latitude/longitude/geocode terminology or recognizable place values.

## Important Distinctions
Distinguish location labels from identifiers, organizational regions, cloud regions, and free-form addresses.

## Quality Expectations
Geographic hierarchy and coordinate ranges should be coherent when applicable.

## Possible Data Quality Problems
Spelling variants, abbreviations, inconsistent hierarchy, invalid coordinates, mixed codes/names.

## Diagnostic Evidence
Check distributions, frequencies, samples, pattern consistency, domain context, related columns, repeated occurrences, units, and whether an apparent anomaly is reproducible.

## Cross-Column Relationships
Country-state-city and latitude-longitude relationships may provide validation.

## Cleaning Decision Guidance
Standardize place names only when equivalence is reliable; do not force ambiguous organizational/cloud regions into geography.

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
