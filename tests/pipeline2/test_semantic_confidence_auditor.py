import unittest

import pandas as pd

from benchmarks.tabarena.semantic_confidence_auditor import (
    audit_state,
    competing_interpretations,
    lexical_name_signal,
    value_pattern_signal,
    value_shape_signal,
)


class SemanticConfidenceAuditorTests(unittest.TestCase):
    def signals(self, name, series, claimed):
        lexical = lexical_name_signal(name, claimed)
        shape = value_shape_signal(series, claimed)
        pattern = value_pattern_signal(series, claimed)
        competing = competing_interpretations(shape, lexical, pattern)
        return lexical, shape, pattern, competing

    def test_binary_numeric_values_are_ambiguous_not_boolean_proof(self):
        signals = self.signals("nr_01", pd.Series([0, 1, 0, 1]), "boolean")
        self.assertFalse(signals[1]["uniquely_supports_claim"])
        self.assertGreater(signals[3]["count"], 1)
        self.assertEqual(audit_state("boolean", 1.0, *signals)[0], "AMBIGUOUS")

    def test_transparent_age_name_supports_measurement(self):
        signals = self.signals("age", pd.Series([21, 35, 52]), "numeric_measurement")
        self.assertEqual(audit_state("numeric_measurement", 0.9, *signals)[0], "SUPPORTED")

    def test_enumerated_labels_contradict_high_confidence_text(self):
        signals = self.signals("LET_IS", pd.Series(["alive", "dead", "alive"]), "text")
        self.assertEqual(
            audit_state("text", 0.95, *signals)[0],
            "UNSUPPORTED_HIGH_CONFIDENCE",
        )


if __name__ == "__main__":
    unittest.main()
