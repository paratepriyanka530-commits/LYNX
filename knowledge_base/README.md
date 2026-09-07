# LYNX Knowledge Base V1

This knowledge base supports four reasoning stages: dataset/domain understanding, column semantic understanding, data-quality/issue detection, and cleaning decisions.

The knowledge base provides general knowledge; observed dataset evidence remains authoritative. Cleaning execution must be deterministic and restricted to registered actions.

## Retrieval Scopes
- Dataset understanding: `domains/` plus relevant structural semantics.
- Column understanding: `semantics/core_types/`, `semantics/concepts/`, `semantics/structural_roles/`.
- Issue detection: `data_quality/`, relevant concepts, and relevant domain knowledge.
- Cleaning decisions: `cleaning_methods/`, `decision_rules/`, relevant issue/concept/domain knowledge.

## Safety Philosophy
Detection is not correction. Suspicious is not invalid. Outlier is not error. Missing does not mean impute. Prefer evidence, recoverability, information preservation, uncertainty, and `no_action` when modification is not justified.
