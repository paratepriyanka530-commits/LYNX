# Paper Figure Specification — Frozen LYNX Pipeline 1

## Figure purpose

Create a horizontal, publication-ready system architecture figure showing that
LYNX Pipeline 1 combines AI-assisted understanding and recommendation with
deterministic observation, authorization, execution, and validation. The visual
must make it impossible to read the LLM as directly cleaning data.

**Proposed caption:**

> **Frozen LYNX Pipeline 1 system architecture.** Deterministic profiling and
> issue detection produce controlled evidence for RAG-assisted Qwen semantic
> understanding and bounded Tier-2 recommendations. Deterministic contradiction,
> readiness, risk, registry, and compilation gates alone authorize allow-listed
> instructions, which execute on copied data and are independently validated
> before audited outputs are written.

## Canvas and visual treatment

- **Orientation:** landscape, left-to-right.
- **Target size:** approximately 180 mm × 82 mm, readable across a two-column
  IEEE page; keep all text legible at final size (approximately 7.5–9 pt).
- **Background:** warm white/light paper (`#FCFBF7`).
- **Linework:** dark navy-black (`#17253A`), 1.1–1.4 pt, slightly imperfect
  hand-drawn box and arrow contours. Use restrained irregularity; no sketch
  texture behind text.
- **Typography:** a clean handwritten-style face with high technical legibility;
  use semibold layer titles and regular compact component labels. If unavailable,
  use a humanist sans-serif rather than a decorative script.
- **Accents:** minimal pastel fills with sufficient monochrome contrast:
  observation pale blue, AI pale lavender, decision pale amber, safety gates pale
  coral, execution pale mint, validation/output pale teal.
- **No decoration:** no mascots, people, robots, gears, timelines, frontend,
  Pipeline 2, database, cloud, or generic “magic AI” imagery.

## Exact top-level structure

Use seven architectural columns across the center band:

1. **INPUT & CONTROL**
2. **OBSERVE [D]**
3. **UNDERSTAND [AI/G]**
4. **DECIDE [D/AI]**
5. **AUTHORIZE [G]**
6. **EXECUTE [D]**
7. **VALIDATE & OUTPUT [G/D]**

Place thin vertical dotted separators between columns. Across the top of the
main band, place the small boundary statement:

> **OBSERVE ≠ UNDERSTAND ≠ DECIDE ≠ AUTHORIZE ≠ EXECUTE ≠ VALIDATE**

## Main horizontal data path

Draw a single dominant navy arrow spine from left to right. Label transitions
with the actual artifacts shown below.

### Column 1 — INPUT & CONTROL

Outer group label: **INPUT & CONTROL**

Stack two small boxes:

1. **File Validation [G]**  
   `existence • format • readability • non-empty`
2. **Load Original DataFrame [D]**

At far left, an input sheet shape:

**RAW DATASET**  
`CSV • XLSX • XLS`

Place a small lock annotation below:

`source path protected • raw file never overwritten`

Artifact arrow to Column 2: **Original DataFrame**.

Show **Pipeline Controller [D]** as a narrow dark-outline rail running beneath
all seven columns, labelled `run_pipeline1 • routing • ordering • report assembly`.
It coordinates components but must not obscure the responsibility boundaries.

### Column 2 — OBSERVE [D]

Pastel blue group containing:

- **Column + Dataset Profiler**  
  `shape • dtype • nulls • uniqueness • repetition • samples • statistics`
- **Issue Detector**  
  `issues • candidates • observations`
- small attached tab: **Quality Summary**

Use two outgoing artifact arrows:

- **Dataset Profile** → Understand
- **DetectedIssue[]** → Decide

No arrow from Issue Detector directly to Execute.

### Column 3 — UNDERSTAND [AI/G]

Pastel lavender group with three internal stages:

1. **Evidence Builder [D]**  
   `controlled profile + bounded samples`
2. **Architecture B Semantics [AI]**
   - `Dataset context: row entity • domain • description`
   - `Column batches ≤12: type • value function • numeric meaning`
   - small gate badge: `exact-name coverage • retry • unknown fallback`
3. **Contradiction Guard [G]**  
   `direct value contradiction only`  
   small sublabel: `raw + effective semantics retained`

Artifact arrow to Decide: **Effective SemanticProfile + Audit**.

Add a small “no override” note under the guard:

`ambiguous / binary / low-cardinality numeric shape → unchanged`

### Column 4 — DECIDE [D/AI]

Pastel amber group split vertically into two lanes that rejoin:

**Tier 1 [D]**

- `trim whitespace`
- `rare category → no_action`
- `outlier / duplicate → review`
- `protected/unknown missingness → review or insufficient evidence`

**Tier 2 [AI recommendation]**

- `general issue batches ≤4`
- `strict missing-value batches ≤4`
- small recovery badge: `salvage valid siblings • retry unresolved • fallback`

At the join, use a document shape:

**CleaningDecision[]**  
`lineage • confidence • risk • requires_review • source`

Put the word **RECOMMENDATION** beside the Tier-2-to-decision arrow, never
“action” or “command.”

### Column 5 — AUTHORIZE [G]

This must be the visually strongest safety column. Use pale coral, a double-line
border, and three compact stacked gates:

1. **Risk / Review / Alignment Gate**  
   `invalid response → reject/retry/fallback`
2. **Missing Median Dual Gate**  
   left mini-shield: `semantic eligibility`  
   plus sign  
   right mini-shield: `mechanical preflight`  
   sublabel: `identifier/integer-code risk can veto`
3. **Instruction Compiler + Registry**  
   `targets • parameters • dtype • coverage • semantics`

Beside Gate 3, show a small allow-list card:

**Registered handlers only**

`no_op`  
`trim whitespace`  
`exact category mapping`  
`parse date/time`  
`coerce numeric`  
`impute missing`

Artifact arrow to Execute: **Approved CleaningInstruction[] only**.

Draw rejected/review branches downward from Gates 1–3 into a small tray:

**Preserve / Skip / Review**  
`no_action • insufficient_evidence • flag_for_review`

This tray feeds the audit report, not the executor.

### Column 6 — EXECUTE [D]

Pastel mint group:

- **Deep-copy Deterministic Executor**
- `ordered registered handlers`
- `handler failure → pre-instruction copy`
- small audit tag: `changed cells • examples • lineage • status`

Show the **Original DataFrame** entering a copy/fork symbol before this box.
The original branch continues untouched to post-clean validation; only the copy
enters the executor.

Artifact arrow to Column 7: **Working Cleaned DataFrame + ExecutionResult[]**.

### Column 7 — VALIDATE & OUTPUT [G/D]

Pastel teal group:

1. **Post-clean Validator [G]**  
   `reprofile • redetect • replay approved changes`  
   `scope • rows/columns • dtype • values • nulls • categories`
2. fork to two right-edge output shapes:
   - **cleaned_dataset.csv**
   - **cleaning_report.json**

Show the untouched **Original DataFrame** entering the validator as a second,
thin comparison arrow from below.

Attach a report contents note:

`profiles • issues • raw/effective semantics • decisions • skips`  
`instructions • executions • validation • warnings • timings`

Add a small status badge between validator and outputs:

`validation determines success / failed status`

Do not imply that a failed status is silently discarded: both artifacts remain
inspectable.

## Supporting components

### Upper support band: Knowledge Base / RAG

Above the Understand column, draw a lavender-outline side subsystem:

**LOCAL KNOWLEDGE BASE + RAG [AI]**

`Markdown + YAML metadata`  
→ `H2 chunks + provenance`  
→ `MiniLM embeddings`  
→ `in-memory cosine search`

Draw arrows only to:

- Dataset Context inference
- Batched Column inference

Label the arrows **general semantic context, not dataset fact**. Do not connect
RAG directly to Decide, Authorize, or Execute.

### Upper support band: Ollama / Qwen

Above and between Understand and Decide, draw:

**OLLAMA + QWEN 2.5:3B [AI]**  
`local /api/generate • temperature 0 • structured JSON`

Two downward arrows:

- to **Architecture B Semantics**, labelled `semantic inference`
- to **Tier 2**, labelled `recommendations only`

Terminate both arrows before Authorize. Add a tiny barred-arrow annotation from
the Ollama box toward Execute:

`NO execution path`

### Lower control band: schemas and provenance

Above the controller rail or immediately below the main band, draw a thin
control rail:

**PYDANTIC CONTRACTS [G]**  
`SemanticProfile • CleaningDecision • CleaningInstruction • ExecutionResult • ValidationReport`

Use dotted control arrows to Understand, Decide, Authorize, and Validate.

Beside it draw:

**AUDIT / TELEMETRY**  
`LLM timing + batches + retries + recovery + contradiction + authorization + execution`

Use dotted collection arrows from Understand, Decide, Authorize, Execute, and
Validate into the final JSON report.

## Exact arrow and object labels

Use these labels verbatim where space permits:

```text
Raw Dataset
Original DataFrame
Dataset Profile
Dataset Semantic Evidence
DetectedIssue[]
Raw SemanticProfile
Effective SemanticProfile + contradiction audit
CleaningDecision[]
Approved CleaningInstruction[]
ExecutionResult[] + Cleaned DataFrame
ValidationReport
cleaned_dataset.csv
cleaning_report.json
```

## Safety Controls side panel

Place a compact panel at the bottom-right or along the lower margin. Use shield
bullets, not paragraphs:

- **Raw preservation:** source never overwritten; execution uses deep copies.
- **AI isolation:** structured understanding/recommendations; no generated code.
- **Allow-list:** unknown actions cannot compile or execute.
- **Median dual gate:** recommendation + semantic eligibility + mechanical check.
- **Identifier veto:** suspicious numeric labels can block unsafe magnitude use.
- **Narrow semantic guard:** direct contradiction only; ambiguity does not override.
- **Failure isolation:** salvage valid siblings; retry/fallback unresolved issues.
- **Conservative outcomes:** preserve, insufficient evidence, or review are valid.
- **Independent validation:** approved changes are replayed and compared before output.

Keep this panel to no more than two short lines per bullet at final size.

## Legend

Place a small legend in the lower-left:

- **[D] Deterministic** — solid pale blue/mint fill
- **[AI] LLM/RAG-assisted** — pale lavender fill
- **[G] Deterministic safety gate** — pale coral fill with double-line border
- **Solid arrow** — data/artifact flow
- **Dotted arrow** — control, validation, or audit relationship

## Layout wireframe

```text
          ┌──── LOCAL KNOWLEDGE BASE + RAG [AI] ────┐   ┌─ OLLAMA / QWEN 2.5:3B [AI] ─┐
          │ Markdown → chunks → MiniLM → vector search│   │ semantics + recommendations │
          └─────────────────┬─────────────────────────┘   └──────┬───────────┬──────────┘
                            │ general context                     │           │
                            ▼                                     ▼           ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ╔══════════════════╗  ┌──────────────┐  ┌──────────────────┐
│ INPUT &     │  │ OBSERVE [D]  │  │ UNDERSTAND       │  │ DECIDE       │  ║ AUTHORIZE [G]    ║  │ EXECUTE [D]  │  │ VALIDATE + OUTPUT│
│ CONTROL     │→ │ profile      │→ │ evidence         │→ │ Tier 1 [D]   │→ ║ risk/alignment   ║→ │ deep-copy    │→ │ post-clean gate  │
│ validate    │  │ issues       │  │ Arch B [AI]      │  │ Tier 2 [AI]  │  ║ median dual gate ║  │ registered   │  │ cleaned CSV      │
│ load raw    │  │ summary      │  │ contradiction [G]│  │ recovery     │  ║ compiler/registry║  │ handlers     │  │ JSON audit report│
└─────────────┘  └──────────────┘  └──────────────────┘  └──────────────┘  ╚══════════════════╝  └──────────────┘  └──────────────────┘
       │                                                                                      ▲             ▲
       └──────────── untouched Original DataFrame comparison ────────────────────────────────┘             │
                                                                  Preserve / Skip / Review ─────→ JSON audit

──────────────────────── Pipeline Controller [D] • ordering • routing • report assembly ─────────────────────
──────────── Pydantic Contracts [G] ─────────────── Audit / Telemetry dotted collection rail ────────────────
```

## Accuracy constraints for the illustrator

1. Do not draw standalone `column_understanding.py` or role resolution; neither
   participates in frozen production Architecture B.
2. Do not label Qwen as Qwen 7B. Production defaults use `qwen2.5:3b`.
3. Do not show the LLM producing `CleaningInstruction` or touching a DataFrame.
4. Do not show a database or separate semantic-output file. Semantic/profile
   metadata is nested in `cleaning_report.json`.
5. Do not imply every registry action is automatically reachable. The figure
   must place risk/review and compiler gates before the registry/executor path.
6. Do not depict `flag_rows`, `add_missing_indicator`, or duplicate-row deletion
   as executable frozen handlers. They are not in the action registry.
7. Do not depict post-clean validation as a cosmetic report step; it independently
   checks the cleaned frame against the original and approved instructions.
8. Do not show Pipeline 2, benchmark execution, frontend services, or database
   repositories.

## Central visual message

The final image should read, even at thumbnail scale, as:

> **RAW DATA → DETERMINISTIC OBSERVATION → SEMANTIC UNDERSTANDING [AI] → HYBRID
> DECISION → DETERMINISTIC AUTHORIZATION [G] → DETERMINISTIC EXECUTION →
> VALIDATED CLEAN DATA + AUDIT**

The coral authorization column and the absence of any Ollama-to-executor arrow
are the two most important visual cues.
