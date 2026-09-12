# Pipeline 1 benchmark freeze

Pipeline 1 (`pipeline1_architecture_b_v001`) is frozen for the upcoming
benchmark as of 2026-09-10.

The final hardening change is the deterministic post-inference contradiction
guard. It preserves every raw LLM semantic result in audit telemetry and only
changes an effective result when full-column values directly contradict it.
Ambiguous binary or low-cardinality numeric shapes never trigger an override.

No further Pipeline 1 prompt, model, schema, semantic, decision, or cleaning
changes are permitted before the benchmark is complete.

## Final verification findings and documented limitations

These findings close the final verification pass. They document limitations and
interpretation boundaries; they do not reopen the frozen architecture or
authorize changes to production code, prompts, schemas, validators, or models.

### Contradiction-guard coverage boundary

The enumerated-string-label rule is deliberately conservative. It requires at
least 20 non-null strings, between 2 and 20 distinct labels, a unique ratio no
greater than 0.05, and **every observed label to occur at least twice**. The last condition
creates a conservative coverage gap: in `splice.csv`, 36 of the 60 nucleotide
position columns contain an otherwise finite one-character code alphabet but
also contain a rare ambiguity symbol such as `N`, `D`, `R`, or `S` exactly
once. Those 36 columns do not produce the enumerated-label signal. The remaining
24 position columns and `SiteType` do produce it. Therefore, if an LLM assigned
`text` to one of the 36 excluded position columns, the current guard would not
correct it even though the full value set is categorical in context.

This is a demonstrated detector-coverage boundary, not a demonstrated
production false negative: the cross-dataset `qwen2.5:7b` spot-check produced no
usable semantic batches because Ollama returned HTTP 500 for the initial
12-column `splice` batch and subsequently stalled on reduced batches. No actual
cross-dataset correction count or false-positive rate was obtained. Deterministic
value scans found 25 potential enumerated-label triggers in `splice`, 1 in
`hiva_agnostic` (`CompoundActivity`), and 14 in `kddcup09_appetency`; all inspected
triggers were bounded code vocabularies for which `text` to `categorical` would
be a defensible correction. No guard rule was expanded after this finding.

### Dataset-context null mechanism

The repeated null `probable_domain` and `dataset_description` values are caused
by empty or omitted optional fields at the validated model-output boundary, not
by a downstream extraction bug and not by a non-null value being silently
dropped after schema validation. The dataset-context response schema calls the
domain field `dataset_domain`; the batched analyzer intentionally maps it to
`BatchedDatasetContext.probable_domain`. A deterministic round-trip with
non-null `dataset_domain` and `dataset_description` preserves both fields.

The completed `dataset_09`, `dataset_13`, and MIC/task-363621 calls produced
positive-confidence, non-null row entities (respectively `survey response`,
`unknown`, and `a blood donation record`) while leaving domain and description
null. A schema-validation failure would instead enter the exception path and
replace the entire dataset context with `row_entity=null` and confidence `0.0`,
which did not occur. Only `confidence` is required by the current
`DatasetSemanticInference` JSON schema; domain, description, row entity, and
reasoning are optional and default to null. Historical raw response text was not
persisted, and a fresh raw-capture call timed out, so explicit JSON `null` cannot
be distinguished from field omission. This is documented as a permissive-schema
and model-output limitation rather than a parsing defect.

### `error_code` decision-taxonomy boundary

The 7B nine-case diagnostic's `error_code` result must not be reported as an
unambiguous model-decision failure. The supplied evidence was 27,679 missing
values out of 30,000 (92.2633%, `very_high`), 2,321 observed strings,
`categorical` / `classification` semantics at confidence 0.8, no identifier or
entity-reference protection, no affirmative preservation evidence, and no
supplied consequence evidence. Median semantic eligibility was false and the
mechanical preflight failed because automatic imputation permits only low
missingness. The model returned `insufficient_evidence/no_op`; that response was
accepted on its first attempt without retry or fallback.

This case exposes a genuine boundary in the four-branch contract. The validator
explicitly permits `flag_for_review/flag_rows` with
`review_basis="high_missingness_requires_review"` for high or very-high
missingness, so the manifest's expected outcome is operationally reasonable.
However, the prompt also directs the model to use `insufficient_evidence` when
imputation is unavailable, affirmative preservation evidence is absent, and no
supplied consequence supports review. Both readings fit the evidence presented.
For paper reporting, classify this case as a **decision-taxonomy boundary
ambiguity**, not a plain semantic-decision mismatch; `flag_for_review` may remain
the preferred operational label while `insufficient_evidence` remains defensible
under the written rubric.
