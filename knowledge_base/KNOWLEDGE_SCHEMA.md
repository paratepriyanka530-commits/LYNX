# LYNX Knowledge Document Standard

Every active knowledge document uses YAML front matter with `knowledge_id`, `category`, `topic`, `version`, and `status`.

## Standard Sections
1. Definition
2. Semantic Indicators
3. Important Distinctions
4. Quality Expectations
5. Possible Data Quality Problems
6. Diagnostic Evidence
7. Cross-Column Relationships
8. Cleaning Decision Guidance
9. When NOT to Clean
10. Evidence Required Before Modification
11. Reasoning Principles

## Global Rules
- Dataset evidence is authoritative for observed facts; retrieved knowledge supplies general guidance.
- Column names are evidence, never proof.
- Physical dtype is not semantic meaning.
- Detection and correction are separate decisions.
- An outlier is not automatically invalid.
- A missing value does not automatically require imputation.
- A rare value is not automatically erroneous.
- Never fabricate a replacement value.
- Prefer the least destructive justified action.
- Use cross-column evidence when semantically valid.
- `no_action`, `flag_for_review`, and `needs_additional_context` are legitimate outcomes.
