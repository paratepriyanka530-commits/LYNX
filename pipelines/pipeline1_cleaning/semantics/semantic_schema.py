from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


SemanticType = Literal[
    "identifier",
    "numeric_measurement",
    "categorical",
    "text",
    "date",
    "time",
    "datetime",
    "boolean",
    "currency",
    "percentage",
    "contact_information",
    "location",
    "unknown",
]


SemanticRole = Literal[
    "primary_identifier",
    "foreign_identifier",
    "transaction_identifier",
    "target_candidate",
    "feature",
    "timestamp",
    "category",
    "measurement",
    "description",
    "contact_field",
    "unknown",
]

ValueFunction = Literal[
    "entity_reference",
    "classification",
    "measurement",
    "temporal",
    "descriptive_attribute",
    "boolean_state",
    "contact",
    "location",
    "unknown",
]

EntityAlignment = Literal[
    "same_as_row_entity",
    "different_from_row_entity",
    "uncertain",
    "not_applicable",
]


class ColumnSemanticResult(BaseModel):
    column_name: str
    semantic_type: SemanticType
    semantic_role: SemanticRole
    value_function: ValueFunction
    concept_description: str = Field(min_length=1)

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    reasoning_summary: str = Field(min_length=1)

    rag_sources: List[str] = Field(
        default_factory=list
    )

class DatasetSemanticResult(BaseModel):
    dataset_domain: Optional[str] = None
    dataset_description: Optional[str] = None
    row_entity: Optional[str] = None

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    reasoning_summary: Optional[str] = None

    rag_sources: List[str] = Field(
        default_factory=list
    )

class ColumnSemanticInference(BaseModel):
    semantic_type: SemanticType
    value_function: ValueFunction
    concept_description: str = Field(min_length=1)

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    reasoning_summary: str = Field(min_length=1)

class DatasetSemanticInference(BaseModel):
    dataset_domain: Optional[str] = None
    dataset_description: Optional[str] = None
    row_entity: Optional[str] = None

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    reasoning_summary: Optional[str] = None

class ColumnRoleInference(BaseModel):
    referenced_entity: Optional[str] = Field(
        description=(
            "The entity or instance concept identified by the column values. "
            "This is not the dataset row entity unless the column actually "
            "identifies that same entity. Use null when the column does not "
            "reference an entity."
        )
    )
    entity_alignment: EntityAlignment = Field(
        description=(
            "The semantic relationship between the entity identified by the "
            "column values and the dataset row entity."
        )
    )
    semantic_role: SemanticRole
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(
        min_length=1,
        description=(
            "Explains the entity identified by the column, the entity or grain "
            "represented by one dataset row, and why those operands support "
            "the selected entity alignment and semantic role."
        ),
    )

    @model_validator(mode="after")
    def validate_entity_alignment(self):
        identifier_roles = {
            "primary_identifier",
            "foreign_identifier",
            "transaction_identifier",
        }

        if self.entity_alignment in {
            "same_as_row_entity",
            "different_from_row_entity",
        } and (
            self.referenced_entity is None
            or not self.referenced_entity.strip()
        ):
            raise ValueError(
                "A resolved entity alignment requires referenced_entity."
            )

        if (
            self.entity_alignment == "not_applicable"
            and self.referenced_entity is not None
        ):
            raise ValueError(
                "referenced_entity must be null when alignment is not applicable."
            )

        if (
            self.entity_alignment == "same_as_row_entity"
            and self.semantic_role == "foreign_identifier"
        ):
            raise ValueError(
                "A same-row-entity reference cannot be a foreign identifier."
            )

        if (
            self.entity_alignment == "different_from_row_entity"
            and self.semantic_role == "primary_identifier"
        ):
            raise ValueError(
                "A different-entity reference cannot be a primary identifier."
            )

        if (
            self.entity_alignment == "uncertain"
            and self.semantic_role in identifier_roles
            and self.confidence > 0.9
        ):
            raise ValueError(
                "Uncertain entity alignment cannot support a near-certain "
                "identifier role."
            )

        return self


class ColumnRoleResult(ColumnRoleInference):
    column_name: str
    rag_sources: List[str] = Field(default_factory=list)
