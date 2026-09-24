# Pipeline 1 privacy masking

Pipeline 1 treats privacy masking as a separate concern from data-quality
repair. Deterministic value patterns detect email addresses, grouped telephone
numbers, and SSN/Aadhaar/PAN-like national identifiers. Ambiguous person-name
and contact cases reuse the column semantic result already produced by
Architecture B; privacy detection does not create an additional LLM call.

## Default policy

| PII candidate subtype | Default | Reason |
|---|---|---|
| email | hash | Preserve equality for grouping or joins without exposing the address. |
| phone_number | hash | Preserve equality for grouping or joins without exposing the number. |
| semantic_contact_information | hash | Preserve a stable pseudonymous key when the exact contact pattern is unknown. |
| national_id | redact | National identifiers have no default analytical use and are high impact. |
| semantic_person_name | redact | Person names have no default downstream modeling use. |

Hashing uses deterministic SHA-256 with a LYNX domain separator. It is
pseudonymization, not anonymization: equality and frequency remain visible,
and low-entropy values may remain vulnerable to guessing. Full redaction
replaces each non-null value with [REDACTED]. Both strategies preserve nulls.

The mask_column action is compiler-restricted to deterministic Tier 1
pii_candidate decisions and their subtype default. It executes after
data-quality actions. Audit entries use reason "masked for privacy" and metric
category "privacy"; privacy counts are excluded from quality-fix summaries.
Raw values from masked columns are omitted from detector examples, execution
change examples, and persisted before-profile samples.

## Deliberate scope boundary: internal surrogate identifiers

Internal surrogate identifiers such as employee_id, patient_id, student_id,
user_id, customer_id, account_id, and transaction_id are intentionally not
treated as PII candidates. This is a deliberate scope boundary, not a
detection gap. Sensitivity in this pipeline is defined by what a field directly
reveals or enables: contact information, government-issued identifiers, or
personal names. A surrogate identifier is a pointer that does not itself expose
contact details or create direct impersonation or fraud risk, and it may be
needed for grouping, joins, longitudinal analysis, or legitimate predictive
use.

Re-identification risk from these identifiers depends on whether they are
joined against identity-mapping data outside this pipeline's control. Such
external linkage is outside the current masking policy. Consequently,
person-linked and event/object-linked surrogate identifiers are both retained;
the detector does not infer privacy sensitivity merely from an identifier
semantic role or an *_id-style column name.

## Pipeline 2 handoff enforcement

The Pipeline 2 benchmark runner accepts only a per-dataset Pipeline 1 artifact
directory. A dataset is eligible only when its report records successful
Pipeline 1 and post-clean validation, an explicitly completed and passed
privacy stage, complete successful mask audit coverage for every PII candidate,
and a SHA-256 matching the cleaned CSV. Missing or invalid artifacts fail
closed; the runner has no raw-input fallback.
