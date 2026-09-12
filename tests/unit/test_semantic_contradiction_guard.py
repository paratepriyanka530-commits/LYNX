import unittest
import tempfile
from pathlib import Path

import pandas as pd

from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnSemantic,
    BatchedDatasetContext,
    SemanticProfile,
)
from pipelines.pipeline1_cleaning.semantics.contradiction_guard import (
    CORRECTION_CONFIDENCE_CAP,
    apply_semantic_contradiction_guard,
)
from pipelines.pipeline1_cleaning.pipeline import run_pipeline1


ROOT = Path(__file__).resolve().parents[2]
MIC = ROOT / "benchmarks" / "tabarena" / "datasets" / "MIC.csv"


def semantic(name: str, semantic_type: str, confidence: float = 1.0) -> BatchedColumnSemantic:
    mappings = {
        "text": ("descriptive_attribute", "not_applicable"),
        "categorical": ("classification", "not_applicable"),
        "boolean": ("boolean_state", "nominal_code"),
        "numeric_measurement": ("measurement", "magnitude"),
    }
    value_function, numeric_meaning = mappings[semantic_type]
    return BatchedColumnSemantic(
        column_name=name,
        semantic_type=semantic_type,
        value_function=value_function,
        numeric_meaning=numeric_meaning,
        concept_description="Original LLM description.",
        semantic_role="unknown",
        confidence=confidence,
        reasoning_summary="Original LLM reasoning.",
    )


def apply(frame: pd.DataFrame, *columns: BatchedColumnSemantic) -> SemanticProfile:
    profile = SemanticProfile(
        dataset_context=BatchedDatasetContext(row_entity=None, row_entity_confidence=0.0),
        columns=list(columns),
        telemetry={"total_llm_calls": 1},
    )
    return apply_semantic_contradiction_guard(profile, frame)


class SemanticContradictionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mic = pd.read_csv(
            MIC,
            usecols=["LET_IS", "AGE", "SEX", "nr_01", "ritm_ecg_p_01"],
            low_memory=False,
        )

    def test_mic_required_columns_only_correct_direct_contradiction(self):
        originals = [
            semantic("LET_IS", "text", 0.95),
            semantic("AGE", "numeric_measurement"),
            semantic("SEX", "categorical"),
            semantic("nr_01", "categorical", 0.95),
            semantic("ritm_ecg_p_01", "boolean"),
        ]
        result = apply(self.mic, *originals)
        effective = {column.column_name: column for column in result.columns}

        self.assertEqual(effective["LET_IS"].semantic_type, "categorical")
        self.assertEqual(effective["LET_IS"].value_function, "classification")
        self.assertLessEqual(effective["LET_IS"].confidence, CORRECTION_CONFIDENCE_CAP)
        for name in ("AGE", "SEX", "nr_01", "ritm_ecg_p_01"):
            self.assertEqual(effective[name], next(item for item in originals if item.column_name == name))

        audit = result.telemetry["semantic_contradiction_guard"]
        self.assertEqual(audit["correction_count"], 1)
        let_audit = next(item for item in audit["entries"] if item["column_name"] == "LET_IS")
        self.assertEqual(let_audit["original_llm_result"]["semantic_type"], "text")
        self.assertEqual(let_audit["original_llm_result"]["confidence"], 0.95)
        self.assertEqual(let_audit["effective_semantic_result"]["semantic_type"], "categorical")

    def test_binary_numeric_shape_never_forces_override(self):
        frame = pd.DataFrame({"nr_01": [0, 1] * 50, "ritm_ecg_p_01": [1, 0] * 50})
        originals = [semantic("nr_01", "categorical"), semantic("ritm_ecg_p_01", "boolean")]
        result = apply(frame, *originals)
        self.assertEqual(result.columns, originals)
        self.assertEqual(result.telemetry["semantic_contradiction_guard"]["correction_count"], 0)

    def test_source_port_numeric_protection_is_unaffected(self):
        frame = pd.DataFrame({"source_port": list(range(1000, 1100))})
        original = semantic("source_port", "numeric_measurement")
        result = apply(frame, original)
        self.assertEqual(result.columns[0], original)

    def test_explicit_dates_correct_numeric_measurement_using_existing_parser(self):
        frame = pd.DataFrame({"when": [f"2026-09-{day:02d}" for day in range(1, 21)]})
        result = apply(frame, semantic("when", "numeric_measurement", 0.99))
        self.assertEqual(result.columns[0].semantic_type, "date")
        self.assertEqual(result.columns[0].value_function, "temporal")

    def test_uuid_email_and_url_patterns_are_narrow_corrections(self):
        frame = pd.DataFrame({
            "key": [f"123e4567-e89b-12d3-a456-{index:012d}" for index in range(3)],
            "mail": [f"person{index}@example.com" for index in range(3)],
            "site": [f"https://example.com/{index}" for index in range(3)],
        })
        result = apply(
            frame,
            semantic("key", "numeric_measurement"),
            semantic("mail", "numeric_measurement"),
            semantic("site", "numeric_measurement"),
        )
        self.assertEqual(
            [column.semantic_type for column in result.columns],
            ["identifier", "contact_information", "unknown"],
        )

    def test_small_or_nonrepeated_string_sets_do_not_trigger_text_override(self):
        frame = pd.DataFrame({"note": ["open", "closed", "open"]})
        original = semantic("note", "text", 0.99)
        self.assertEqual(apply(frame, original).columns[0], original)

    def test_pipeline_report_contains_raw_and_effective_semantics(self):
        frame = pd.DataFrame({
            "LET_IS": (["alive", "deceased"] * 20),
            "AGE": list(range(20, 60)),
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            output_path = Path(temp_dir) / "output"
            frame.to_csv(input_path, index=False)

            def mock_semantics(profile, evidence):
                return SemanticProfile(
                    dataset_context=BatchedDatasetContext(
                        row_entity=None, row_entity_confidence=0.0,
                    ),
                    columns=[
                        semantic("LET_IS", "text", 0.95),
                        semantic("AGE", "numeric_measurement", 1.0),
                    ],
                )

            report = run_pipeline1(
                input_path,
                output_path,
                semantic_analyzer=mock_semantics,
                decision_engine=lambda issues, semantics: ([], {}, []),
            )

        effective = {
            item["column_name"]: item
            for item in report.semantic_profile["columns"]
        }
        self.assertEqual(effective["LET_IS"]["semantic_type"], "categorical")
        self.assertEqual(effective["AGE"]["semantic_type"], "numeric_measurement")
        audit = report.semantic_profile["telemetry"]["semantic_contradiction_guard"]
        let_audit = next(item for item in audit["entries"] if item["column_name"] == "LET_IS")
        self.assertEqual(let_audit["original_llm_result"]["semantic_type"], "text")
        self.assertEqual(let_audit["effective_semantic_result"]["semantic_type"], "categorical")


if __name__ == "__main__":
    unittest.main()
