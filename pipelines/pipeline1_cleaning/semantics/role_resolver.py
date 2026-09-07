from typing import Any, Dict, List

from intelligence.llm.provider import LLMProvider
from intelligence.rag.retriever import RAGRetriever

from pipelines.pipeline1_cleaning.semantics.semantic_schema import (
    ColumnRoleInference,
    ColumnRoleResult,
    ColumnSemanticResult,
    DatasetSemanticResult,
)


def build_role_query(
    column_evidence: Dict[str, Any],
    column_semantics: ColumnSemanticResult,
    dataset_semantics: DatasetSemanticResult,
) -> str:
    return f"""
Determine the structural role of one already-understood column.

Dataset row entity:
{dataset_semantics.row_entity}

Dataset domain:
{dataset_semantics.dataset_domain}

Column name:
{column_evidence.get("column_name")}

Value function:
{column_semantics.value_function}

Concept description:
{column_semantics.concept_description}

Physical dtype:
{column_evidence.get("physical_dtype")}

Unique ratio:
{column_evidence.get("uniqueness", {}).get("unique_ratio")}

Repeated value occurrences:
{column_evidence.get("repetition", {}).get("repeated_occurrences")}

Sample values:
{column_evidence.get("sample_values")}

Supporting semantic-type hypothesis:
{column_semantics.semantic_type}
""".strip()


def retrieve_role_context(
    column_evidence: Dict[str, Any],
    column_semantics: ColumnSemanticResult,
    dataset_semantics: DatasetSemanticResult,
    retriever: RAGRetriever,
    top_k: int = 5,
) -> List[Dict[str, Any]]:

    query = build_role_query(
        column_evidence=column_evidence,
        column_semantics=column_semantics,
        dataset_semantics=dataset_semantics,
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        categories=[
            "structural_role",
        ],
    )


def resolve_column_role(
    column_evidence: Dict[str, Any],
    column_semantics: ColumnSemanticResult,
    dataset_semantics: DatasetSemanticResult,
    retriever: RAGRetriever,
    llm: LLMProvider,
    top_k: int = 5,
) -> ColumnRoleResult:

    role_context = retrieve_role_context(
        column_evidence=column_evidence,
        column_semantics=column_semantics,
        dataset_semantics=dataset_semantics,
        retriever=retriever,
        top_k=top_k,
    )

    formatted_context = "\n\n".join(
        f"""
Role topic: {item["topic"]}
Section: {item["section"]}
Knowledge:
{item["content"]}
""".strip()
        for item in role_context
    )

    prompt = f"""
You are resolving the structural role of ONE semantically interpreted dataset column.

Do not change or return the semantic type or value function. Value function
and concept description are the primary semantic context. Treat semantic type
as supporting evidence and a fallible hypothesis, never as a constraint on
structural role. Semantic type and structural role are separate dimensions.
Do not perform data-quality detection.
Do not recommend cleaning.

Observed dataset context is authoritative.

DATASET ROW ENTITY — comparison target:
{dataset_semantics.row_entity}

COLUMN CONCEPT — evidence describing what the values identify:
{column_semantics.concept_description}

VALUE FUNCTION:
{column_semantics.value_function}

Dataset domain:
{dataset_semantics.dataset_domain}

Column name:
{column_evidence.get("column_name")}

Supporting semantic-type hypothesis:
{column_semantics.semantic_type}

Physical dtype:
{column_evidence.get("physical_dtype")}

Unique ratio:
{column_evidence.get("uniqueness", {}).get("unique_ratio")}

Repeated value occurrences:
{column_evidence.get("repetition", {}).get("repeated_occurrences")}

Sample values:
{column_evidence.get("sample_values")}

Relevant structural-role knowledge:
{formatted_context}

Determine how this column functions relative to the dataset's row entity.

Retrieved RAG chunks are definitions and candidate distinctions. Their rank
is NOT evidence that a role applies to this column. Observed semantic alignment
with the row entity is more important than retrieval ordering.

Reason in this order before choosing semantic_role:

1. Inspect value_function and concept_description.
2. Decide whether the values denote a specific entity or instance.
3. If so, determine the entity identified by the COLUMN VALUES and return it
   as referenced_entity.
4. Compare referenced_entity with the dataset row entity.
5. Return entity_alignment.
6. Select semantic_role from that alignment and the surrounding evidence.

OUTPUT FIELD DEFINITIONS

referenced_entity:
The entity identified by the COLUMN VALUES. Derive it from the column concept,
value function, and observed column evidence. Do not copy the dataset row entity
unless the column itself identifies that same entity.

row_entity:
The entity or grain represented by one dataset row. It is the comparison target
and is supplied above; do not return it as referenced_entity merely because it
is the comparison target.

entity_alignment:
The relationship between referenced_entity and row_entity.

Generic comparison examples:

- column identifies entity A; row represents entity A
  -> referenced_entity = A
  -> entity_alignment = same_as_row_entity

- column identifies entity B; row represents entity A
  -> referenced_entity = B
  -> entity_alignment = different_from_row_entity

Use null referenced_entity and "not_applicable" alignment when the column does
not denote a specific entity, including a reusable classification.

Use "uncertain" when the referenced concept or its relationship to row grain
cannot be supported. Uncertain alignment must not produce near-certain primary
or foreign identifier confidence.

Apply these general consistency rules:

- entity_reference + same_as_row_entity supports primary_identifier;
- entity_reference + different_from_row_entity supports foreign_identifier;
- classification + not_applicable supports category;
- transaction_identifier remains available when the concept denotes a
  transaction or event identifier and its relationship to row grain supports it.

reasoning_summary must explicitly state both comparison operands and the
conclusion: the column-referenced entity, the dataset row entity, why they
produce entity_alignment, and why that alignment supports semantic_role.

Use the combined evidence to distinguish these value functions:

- entity identity: the value denotes a particular entity or record;
- classification: the value denotes a reusable class, state, type, level,
  taxonomy member, or grouping;
- another structural function represented by the allowed role vocabulary.

Then compare entity identity with the row grain when applicable:

- When value_function is "entity_reference", determine what entity the
  values denote from the concept description and evidence, then compare it
  with the dataset row entity.

- When value_function is "classification", category is normally the fitting
  structural role because the values denote reusable classes rather than
  entity instances. Depart from that only when stronger observed evidence
  contradicts the upstream function interpretation.

- primary_identifier:
  identifies the dataset's main row entity or record grain.

- foreign_identifier:
  identifies a different related entity participating in the row.

- category:
  represents a reusable classification rather than an entity identity.

An upstream "identifier" hypothesis does not limit the answer to identifier
roles. Code-shaped and identifier-shaped values may function as categories.
Conversely, repetition alone does not prove a category because related entity
identifiers can repeat. Decide whether values denote instances or classes.

Do not choose primary_identifier only because values are unique or
because the column name contains "id", "code", "number", or "key".

Compare what the column identifies with the dataset row entity before
selecting the role.

Return the most appropriate structural role.
""".strip()

    inference_data = llm.generate_structured(
        prompt=prompt,
        response_model=ColumnRoleInference,
    )

    inference = ColumnRoleInference(
        **inference_data
    )

    rag_sources = [
        item["chunk_id"]
        for item in role_context
        if "chunk_id" in item
    ]

    return ColumnRoleResult(
        column_name=column_evidence.get("column_name"),
        referenced_entity=inference.referenced_entity,
        entity_alignment=inference.entity_alignment,
        semantic_role=inference.semantic_role,
        confidence=inference.confidence,
        reasoning_summary=inference.reasoning_summary,
        rag_sources=rag_sources,
    )
