from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from pipelines.pipeline3_results.nlp.schemas import (
    ColumnCandidate,
    ExtractedEntities,
    QueryFilter,
    QueryInterpretation,
    StructuredQuery,
    TermResolution,
)


class SchemaTests(unittest.TestCase):
    def test_structured_query_preserves_terms_without_resolutions(self) -> None:
        query = StructuredQuery(
            raw_text="show average salary by department",
            normalized_text="show average salary by department",
            intent="VISUALIZATION",
            measure_terms=["salary"],
            dimension_terms=["department"],
            aggregation="MEAN",
        )
        self.assertEqual(query.measure_terms, ["salary"])
        self.assertEqual(query.dimension_terms, ["department"])
        self.assertEqual(query.resolved_measures, [])
        self.assertEqual(query.resolved_dimensions, [])
        self.assertEqual(query.comparison, "NONE")

    def test_structured_query_can_add_later_resolution_evidence(self) -> None:
        candidate = ColumnCandidate(
            column_name="total_price",
            score=0.85,
            evidence_sources=["semantic_description", "alias"],
            semantic_evidence={
                "semantic_type": "currency",
                "detected_patterns": ["currency_symbol", "decimal_values"],
            },
            analytical_role="MEASURE",
        )
        resolution = TermResolution(
            term="sales",
            kind="MEASURE",
            candidates=[candidate],
            resolved_column="total_price",
            confidence=0.85,
            reason="one supported candidate",
        )
        query = StructuredQuery(
            raw_text="total sales",
            measure_terms=["sales"],
            aggregation="SUM",
            resolved_measures=["total_price"],
            resolutions=[resolution],
        )
        self.assertEqual(query.measure_terms, ["sales"])
        self.assertEqual(query.resolved_measures, ["total_price"])
        self.assertEqual(query.resolutions[0].candidates[0].score, 0.85)
        self.assertEqual(
            query.resolutions[0].candidates[0].semantic_evidence["detected_patterns"],
            ["currency_symbol", "decimal_values"],
        )

    def test_query_filter_preserves_unresolved_and_resolved_forms(self) -> None:
        unresolved = QueryFilter(
            raw_text="in the north region",
            column_term="region",
            operator="EQUALS",
            value_term="north",
        )
        resolved = unresolved.model_copy(update={"resolved_column": "customer_region"})
        self.assertIsNone(unresolved.resolved_column)
        self.assertEqual(resolved.resolved_column, "customer_region")

    def test_extracted_entities_are_dataset_independent(self) -> None:
        entities = ExtractedEntities(
            measure_terms=["revenue"],
            dimension_terms=["products"],
            comparison="TOP_N",
            limit=5,
        )
        self.assertEqual(entities.measure_terms, ["revenue"])
        self.assertEqual(entities.limit, 5)

    def test_interpretation_is_json_safe(self) -> None:
        interpretation = QueryInterpretation(
            status="NEEDS_CLARIFICATION",
            query=StructuredQuery(
                raw_text="sales by region",
                intent="VISUALIZATION",
                measure_terms=["sales"],
                dimension_terms=["region"],
            ),
            message="I found multiple possible measures for 'sales'.",
            choices=["gross_sales", "net_sales"],
            signals=["two equally supported candidates"],
            confidence=0.5,
        )
        payload = interpretation.model_dump(mode="json")
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["status"], "NEEDS_CLARIFICATION")

    def test_models_are_frozen(self) -> None:
        query = StructuredQuery(raw_text="show revenue")
        with self.assertRaises(ValidationError):
            query.intent = "VISUALIZATION"  # type: ignore[misc]

    def test_confidence_and_limit_constraints(self) -> None:
        with self.assertRaises(ValidationError):
            ColumnCandidate(column_name="sales", score=1.1)
        with self.assertRaises(ValidationError):
            ExtractedEntities(limit=0)

    def test_defaults_are_not_shared_between_instances(self) -> None:
        first = StructuredQuery(raw_text="first")
        second = StructuredQuery(raw_text="second")
        first.measure_terms.append("salary")
        self.assertEqual(second.measure_terms, [])


if __name__ == "__main__":
    unittest.main()
