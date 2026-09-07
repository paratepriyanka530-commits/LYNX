from pprint import pprint

from intelligence.llm.ollama_provider import (
    OllamaProvider,
)

from pipelines.pipeline1_cleaning.semantics.semantic_schema import (
    ColumnSemanticInference,
)

provider = OllamaProvider(
    model="qwen2.5:3b"
)


prompt = """
Return ONLY valid JSON.

Analyze this tabular dataset column.

Column name: order_id
Physical dtype: int64
Sample values: [1, 2, 3, 4]
Unique ratio: 0.44
Repetition rate: 0.83

Dataset columns:
pizza_id,
order_id,
pizza_name,
quantity,
order_date,
order_time,
unit_price

Return exactly these fields:

semantic_type
semantic_role
description
confidence
reasoning_summary
possible_issues

Allowed semantic_type values:
identifier,
numeric_measurement,
categorical,
text,
date,
time,
datetime,
boolean,
currency,
percentage,
contact_information,
location,
unknown

Allowed semantic_role values:
primary_identifier,
foreign_identifier,
transaction_identifier,
target_candidate,
feature,
timestamp,
category,
measurement,
description,
contact_field,
unknown

confidence must be between 0.0 and 1.0.
possible_issues must be a JSON array.
"""


result = provider.generate_structured(
    prompt=prompt,
    response_model=ColumnSemanticInference
)

print(
    "\nOLLAMA RESULT"
)

print("=" * 60)

pprint(
    result
)