"""Architecture B contracts, separate from the reproducible V001 schemas."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from pipelines.pipeline1_cleaning.semantics.semantic_schema import SemanticRole, SemanticType, ValueFunction


NumericMeaning = Literal[
    "magnitude",
    "ordinal",
    "nominal_code",
    "identifier",
    "not_applicable",
    "unknown",
]


class BatchedDatasetContext(BaseModel):
    row_entity: Optional[str] = None
    row_entity_confidence: float = Field(ge=0.0, le=1.0)
    probable_domain: Optional[str] = None
    domain_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dataset_description: Optional[str] = None


class BatchedColumnSemantic(BaseModel):
    column_name: str = Field(min_length=1)
    semantic_type: SemanticType
    value_function: ValueFunction
    numeric_meaning: NumericMeaning = Field(
        description=(
            "How numeric values function semantically: magnitude when arithmetic differences "
            "and central tendency represent a meaningful quantitative amount, rate, or "
            "measurement; ordinal when order is meaningful but distance need not be; "
            "nominal_code when numeric representations select categories, states, addresses, "
            "or labels and arithmetic is meaningless; identifier when values identify an "
            "instance, entity, or record; not_applicable when the values are not semantically "
            "numeric; unknown when evidence is insufficient."
        ),
    )
    concept_description: str = Field(min_length=1)
    semantic_role: SemanticRole = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1)


class BatchedColumnInference(BaseModel):
    columns: List[BatchedColumnSemantic]


class SemanticProfile(BaseModel):
    architecture: str = "architecture_b_batched"
    dataset_context: BatchedDatasetContext
    columns: List[BatchedColumnSemantic]
    warnings: List[str] = Field(default_factory=list)
    telemetry: Dict[str, Any] = Field(default_factory=dict)
