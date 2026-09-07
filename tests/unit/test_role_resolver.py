import re
import unittest

from pydantic import ValidationError

from pipelines.pipeline1_cleaning.semantics.role_resolver import (
    build_role_query,
    resolve_column_role,
)
from pipelines.pipeline1_cleaning.semantics.semantic_schema import (
    ColumnRoleInference,
    ColumnSemanticResult,
    DatasetSemanticResult,
)


class EmptyRetriever:
    def retrieve(self, query, top_k, categories):
        return []


class AlignmentReasoningLLM:
    """Models the required reasoning contract without using column names."""

    def generate_structured(self, prompt, response_model):
        value_function = re.search(
            r"VALUE FUNCTION:\s*([a-z_]+)", prompt
        ).group(1)
        concept = re.search(
            r"COLUMN CONCEPT — evidence describing what the values identify:"
            r"\s*([^\r\n]+)",
            prompt,
        ).group(1).strip().lower()
        row_entity = re.search(
            r"DATASET ROW ENTITY — comparison target:\s*([^\r\n]+)",
            prompt,
        ).group(1).strip().lower()

        if value_function == "classification":
            referenced_entity = None
            alignment = "not_applicable"
            role = "category"
            confidence = 0.9
        elif concept == "uncertain entity":
            referenced_entity = None
            alignment = "uncertain"
            role = "unknown"
            confidence = 0.45
        elif concept == row_entity:
            referenced_entity = concept
            alignment = "same_as_row_entity"
            role = "primary_identifier"
            confidence = 0.9
        else:
            referenced_entity = concept
            alignment = "different_from_row_entity"
            role = "foreign_identifier"
            confidence = 0.9

        return {
            "referenced_entity": referenced_entity,
            "entity_alignment": alignment,
            "semantic_role": role,
            "confidence": confidence,
            "reasoning_summary": (
                "The referenced concept was compared with the dataset row entity."
            ),
        }


class CapturingAlignmentLLM(AlignmentReasoningLLM):
    def generate_structured(self, prompt, response_model):
        self.prompt = prompt
        return super().generate_structured(prompt, response_model)


def make_column_semantics(column_name, value_function, concept):
    semantic_type = (
        "categorical" if value_function == "classification" else "unknown"
    )
    return ColumnSemanticResult(
        column_name=column_name,
        semantic_type=semantic_type,
        semantic_role="unknown",
        value_function=value_function,
        concept_description=concept,
        confidence=0.9,
        reasoning_summary="The values denote the supplied semantic concept.",
    )


def make_evidence(column_name, unique_ratio, repeated_occurrences):
    return {
        "column_name": column_name,
        "physical_dtype": "object",
        "uniqueness": {"unique_ratio": unique_ratio},
        "repetition": {"repeated_occurrences": repeated_occurrences},
        "sample_values": ["A-1", "B-2"],
    }


class RoleResolverTests(unittest.TestCase):
    def test_role_schema_describes_entity_operands(self):
        properties = ColumnRoleInference.model_json_schema()["properties"]

        referenced_description = properties["referenced_entity"]["description"]
        alignment_description = properties["entity_alignment"]["description"]
        reasoning_description = properties["reasoning_summary"]["description"]

        self.assertIn("identified by the column values", referenced_description)
        self.assertIn("not the dataset row entity", referenced_description)
        self.assertIn("Use null", referenced_description)
        self.assertIn("column values", alignment_description)
        self.assertIn("dataset row entity", alignment_description)
        self.assertIn("entity identified by the column", reasoning_description)
        self.assertIn("represented by one dataset row", reasoning_description)

    def test_prompt_distinguishes_referenced_entity_from_row_entity(self):
        llm = CapturingAlignmentLLM()
        result = resolve_column_role(
            column_evidence=make_evidence("party_key", 0.4, 60),
            column_semantics=make_column_semantics(
                "party_key", "entity_reference", "participant"
            ),
            dataset_semantics=DatasetSemanticResult(
                row_entity="event", confidence=0.9
            ),
            retriever=EmptyRetriever(),
            llm=llm,
        )

        self.assertEqual(result.referenced_entity, "participant")
        self.assertIn("DATASET ROW ENTITY — comparison target:\nevent", llm.prompt)
        self.assertIn(
            "COLUMN CONCEPT — evidence describing what the values identify:\n"
            "participant",
            llm.prompt,
        )
        self.assertIn(
            "The entity identified by the COLUMN VALUES", llm.prompt
        )
        self.assertIn(
            "Do not copy the dataset row entity\nunless the column itself "
            "identifies that same entity",
            llm.prompt,
        )
        self.assertIn(
            "The entity or grain represented by one dataset row", llm.prompt
        )
        self.assertIn("both comparison operands", llm.prompt)
        self.assertIn("column-referenced entity", llm.prompt)
        self.assertIn("dataset row entity", llm.prompt)

    def test_entity_alignment_precedes_role_selection(self):
        cases = [
            # Required live regressions.
            ("encounter_id", "entity_reference", "encounter", "encounter",
             0.994, 45, "encounter", "same_as_row_entity",
             "primary_identifier"),
            ("patient_id", "entity_reference", "patient", "encounter",
             0.35, 400, "patient", "different_from_row_entity",
             "foreign_identifier"),
            ("diagnosis_code", "classification", "diagnostic classification",
             "encounter", 0.01, 7000, None, "not_applicable", "category"),
            # Generic cases and evidence counterexamples.
            ("record_key", "entity_reference", "order", "order",
             1.0, 0, "order", "same_as_row_entity", "primary_identifier"),
            ("party_key", "entity_reference", "customer", "order",
             0.999, 2, "customer", "different_from_row_entity",
             "foreign_identifier"),
            ("state_value", "classification", "processing status", "order",
             0.03, 900, None, "not_applicable", "category"),
            ("group_key", "entity_reference", "shipment", "shipment",
             0.4, 600, "shipment", "same_as_row_entity",
             "primary_identifier"),
            ("opaque_key", "entity_reference", "uncertain entity", "event",
             0.8, 20, None, "uncertain", "unknown"),
        ]

        for (column_name, value_function, concept, row_entity, unique_ratio,
             repeated_occurrences, expected_entity, expected_alignment,
             expected_role) in cases:
            with self.subTest(column_name=column_name):
                result = resolve_column_role(
                    column_evidence=make_evidence(
                        column_name, unique_ratio, repeated_occurrences
                    ),
                    column_semantics=make_column_semantics(
                        column_name, value_function, concept
                    ),
                    dataset_semantics=DatasetSemanticResult(
                        row_entity=row_entity, confidence=0.9
                    ),
                    retriever=EmptyRetriever(),
                    llm=AlignmentReasoningLLM(),
                )

                self.assertEqual(result.referenced_entity, expected_entity)
                self.assertEqual(result.entity_alignment, expected_alignment)
                self.assertEqual(result.semantic_role, expected_role)
                if expected_alignment == "uncertain":
                    self.assertLessEqual(result.confidence, 0.9)

    def test_same_entity_cannot_be_foreign_identifier(self):
        with self.assertRaises(ValidationError):
            ColumnRoleInference(
                referenced_entity="record",
                entity_alignment="same_as_row_entity",
                semantic_role="foreign_identifier",
                confidence=0.95,
                reasoning_summary="Contradictory output.",
            )

    def test_different_entity_cannot_be_primary_identifier(self):
        with self.assertRaises(ValidationError):
            ColumnRoleInference(
                referenced_entity="participant",
                entity_alignment="different_from_row_entity",
                semantic_role="primary_identifier",
                confidence=0.95,
                reasoning_summary="Contradictory output.",
            )

    def test_uncertain_alignment_rejects_near_certain_identifier_role(self):
        with self.assertRaises(ValidationError):
            ColumnRoleInference(
                referenced_entity=None,
                entity_alignment="uncertain",
                semantic_role="foreign_identifier",
                confidence=0.99,
                reasoning_summary="The relationship is unresolved.",
            )

    def test_resolved_alignment_requires_referenced_entity(self):
        with self.assertRaises(ValidationError):
            ColumnRoleInference(
                referenced_entity=None,
                entity_alignment="same_as_row_entity",
                semantic_role="primary_identifier",
                confidence=0.8,
                reasoning_summary="Missing referenced concept.",
            )

    def test_role_query_keeps_semantic_contract_and_observed_evidence(self):
        query = build_role_query(
            column_evidence=make_evidence("record_key", 0.8, 20),
            column_semantics=make_column_semantics(
                "record_key", "entity_reference", "record"
            ),
            dataset_semantics=DatasetSemanticResult(
                row_entity="record", confidence=0.9
            ),
        )

        self.assertIn("Value function:\nentity_reference", query)
        self.assertIn("Concept description:\nrecord", query)
        self.assertIn("Dataset row entity:\nrecord", query)
        self.assertIn("Repeated value occurrences:\n20", query)


if __name__ == "__main__":
    unittest.main()
