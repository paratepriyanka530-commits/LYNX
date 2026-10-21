"""Tests for Pipeline 3 Visualization API endpoints.

Covers:
- GET /api/visualizations/{dataset_id} (automatic mode) -> 200, exact shape
- POST /api/visualizations/{dataset_id} (automatic mode alias) -> 200
- POST /api/visualizations/{dataset_id}/query (requested mode) -> 200 for valid queries
- POST /api/visualizations/{dataset_id}/query -> 200 for NEEDS_CLARIFICATION & INVALID
- Unknown dataset_id -> 404
- Malformed request body -> 400 (invalid json, missing field, non-string field, empty string)
- Output structure matches build_visualization_results exactly
- Reuses real datasets from Stage 10 validation (pizza_sales.csv, TabArena 363612)
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api.server import create_app
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VisualizationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    # -----------------------------------------------------------------------
    # 1. Automatic Mode Tests
    # -----------------------------------------------------------------------
    def test_automatic_visualization_tabarena_200(self):
        """GET /api/visualizations/{dataset_id} on TabArena dataset returns 200."""
        resp = self.client.get("/api/visualizations/363612")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["mode"], "automatic")
        self.assertEqual(data["status"], "READY")
        self.assertEqual(data["dataset_name"], "tabarena_363612")
        self.assertGreater(data["visualization_count"], 0)
        self.assertEqual(len(data["visualizations"]), data["visualization_count"])
        self.assertIn("kpis", data)
        self.assertLessEqual(len(data["kpis"]), 4)
        self.assertIn("dashboard", data)

        # Check required fields on each visualization item
        for viz in data["visualizations"]:
            self.assertIn("rank", viz)
            self.assertIn("chart_type", viz)
            self.assertIn("x_column", viz)
            self.assertIn("y_column", viz)
            self.assertIn("score", viz)
            self.assertIsNotNone(viz["score"])  # automatic score preserved
            self.assertIn("chart_data", viz)
            self.assertIn("signals", viz)
            self.assertIn("explanation", viz)
            self.assertIn("visualization_plan", viz)
            self.assertIn("chart_spec", viz)
            self.assertNotIn("xAxis", viz["chart_spec"])
            self.assertNotIn("yAxis", viz["chart_spec"])
            expl = viz["explanation"]
            self.assertIn("why_this_chart", expl)
            self.assertIn("what_it_shows", expl)
            self.assertIn("aggregation", expl)
            self.assertIn("data_coverage", expl)
            self.assertIn("evidence", expl)

    def test_automatic_visualization_domain_200(self):
        """GET /api/visualizations/{dataset_id} on domain dataset returns 200."""
        resp = self.client.get("/api/visualizations/pizza_sales")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["mode"], "automatic")
        self.assertEqual(data["status"], "READY")
        self.assertEqual(data["dataset_name"], "pizza_sales")
        self.assertGreater(len(data["visualizations"]), 0)

    def test_automatic_visualization_post_method_supported(self):
        """POST /api/visualizations/{dataset_id} works as an alias for automatic mode."""
        resp = self.client.post("/api/visualizations/pizza_sales")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "automatic")
        self.assertEqual(data["status"], "READY")

    # -----------------------------------------------------------------------
    # 2. Requested Mode Tests (Valid Queries)
    # -----------------------------------------------------------------------
    def test_requested_visualization_pizza_median_bar(self):
        """POST /api/visualizations/pizza_sales/query with Median query returns 200."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={"request_text": "Median unit_price by pizza_category"},
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["mode"], "requested")
        self.assertEqual(data["request_text"], "Median unit_price by pizza_category")
        self.assertEqual(data["status"], "READY")
        self.assertEqual(data["visualization_count"], 1)
        self.assertEqual(len(data["visualizations"]), 1)

        viz = data["visualizations"][0]
        self.assertIsNone(viz["score"])  # score must be None for requested
        self.assertEqual(viz["chart_type"], "BAR")
        self.assertEqual(viz["x_column"], "pizza_category")
        self.assertEqual(viz["y_column"], "unit_price")
        self.assertEqual(viz["chart_data"]["aggregation"], "median")
        self.assertIn("explanation", viz)
        self.assertEqual(viz["source"], "USER_REQUESTED")
        self.assertEqual(viz["chart_spec"]["family"], "BAR")
        self.assertIn("user-requested", viz["explanation"]["aggregation"].lower())

    def test_requested_visualization_tabarena_scatter(self):
        """POST /api/visualizations/363612/query with scatter query returns 200."""
        resp = self.client.post(
            "/api/visualizations/363612/query",
            json={"request_text": "Scatter plot of attack-angle and scaled-sound-pressure"},
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["mode"], "requested")
        self.assertEqual(data["status"], "READY")
        viz = data["visualizations"][0]
        self.assertEqual(viz["chart_type"], "SCATTER")
        self.assertEqual(viz["x_column"], "attack-angle")
        self.assertEqual(viz["y_column"], "scaled-sound-pressure")
        self.assertIsNone(viz["score"])

    def test_requested_visualization_pizza_doughnut(self):
        """The public API preserves the controlled PIE presentation variant."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={
                "request_text": "total revenue by pizza_category in doughnut graph"
            },
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["status"], "READY")
        viz = data["visualizations"][0]
        self.assertEqual(viz["chart_type"], "PIE")
        self.assertEqual(viz["presentation_variant"], "DOUGHNUT")
        self.assertEqual(viz["chart_data"]["presentation_variant"], "DOUGHNUT")

    # -----------------------------------------------------------------------
    # 3. Legitimate Non-Error Outcomes (NEEDS_CLARIFICATION & INVALID -> 200)
    # -----------------------------------------------------------------------
    def test_requested_needs_clarification_returns_200(self):
        """Vague query returns HTTP 200 with status=NEEDS_CLARIFICATION (not an HTTP error)."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={"request_text": "make a chart"},
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["mode"], "requested")
        self.assertEqual(data["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(data["visualization_count"], 0)
        self.assertEqual(data["visualizations"], [])
        self.assertIn("clarification_options", data)
        self.assertGreater(len(data["clarification_options"]), 0)

    def test_requested_invalid_returns_200(self):
        """Incompatible query returns HTTP 200 with status=INVALID (not an HTTP error)."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={"request_text": "scatter plot of pizza_category and pizza_size"},
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["mode"], "requested")
        self.assertEqual(data["status"], "INVALID")
        self.assertEqual(data["visualization_count"], 0)
        self.assertEqual(data["visualizations"], [])
        self.assertIn("errors", data)
        self.assertGreater(len(data["errors"]), 0)

    # -----------------------------------------------------------------------
    # 4. Error Cases (404 and 400)
    # -----------------------------------------------------------------------
    def test_unknown_dataset_id_returns_404_automatic(self):
        """Unknown dataset_id returns HTTP 404 with dataset_not_found error."""
        resp = self.client.get("/api/visualizations/totally_unknown_dataset_xyz")
        self.assertEqual(resp.status_code, 404)
        err = resp.json()
        self.assertIn("error", err)
        self.assertEqual(err["error"]["code"], "dataset_not_found")
        self.assertIn("totally_unknown_dataset_xyz", err["error"]["message"])

    def test_unknown_dataset_id_returns_404_query(self):
        """Unknown dataset_id in query mode returns HTTP 404."""
        resp = self.client.post(
            "/api/visualizations/totally_unknown_dataset_xyz/query",
            json={"request_text": "bar chart of category"},
        )
        self.assertEqual(resp.status_code, 404)
        err = resp.json()
        self.assertIn("error", err)
        self.assertEqual(err["error"]["code"], "dataset_not_found")

    def test_malformed_request_body_invalid_json_returns_400(self):
        """Non-JSON payload returns HTTP 400."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            content=b"{invalid: json:",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)
        err = resp.json()
        self.assertEqual(err["error"]["code"], "malformed_request")

    def test_malformed_request_body_missing_field_returns_400(self):
        """JSON payload missing 'request_text' returns HTTP 400."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={"wrong_field": "something"},
        )
        self.assertEqual(resp.status_code, 400)
        err = resp.json()
        self.assertEqual(err["error"]["code"], "missing_request_text")

    def test_malformed_request_body_non_string_returns_400(self):
        """JSON payload with non-string 'request_text' returns HTTP 400."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={"request_text": 12345},
        )
        self.assertEqual(resp.status_code, 400)
        err = resp.json()
        self.assertEqual(err["error"]["code"], "invalid_request_text")

    def test_malformed_request_body_empty_string_returns_400(self):
        """JSON payload with whitespace-only 'request_text' returns HTTP 400."""
        resp = self.client.post(
            "/api/visualizations/pizza_sales/query",
            json={"request_text": "   "},
        )
        self.assertEqual(resp.status_code, 400)
        err = resp.json()
        self.assertEqual(err["error"]["code"], "invalid_request_text")

    # -----------------------------------------------------------------------
    # 5. Exact Contract Shape Verification against results_service
    # -----------------------------------------------------------------------
    def test_response_matches_results_service_exact_shape(self):
        """API response matches build_visualization_results output structure verbatim."""
        api_resp = self.client.get("/api/visualizations/pizza_sales")
        self.assertEqual(api_resp.status_code, 200)
        api_data = api_resp.json()

        df_pizza = pd.read_csv(_PROJECT_ROOT / "tests" / "datasets" / "valid" / "pizza_sales.csv")
        direct_data = build_visualization_results(df_pizza, request_text=None, dataset_name="pizza_sales")

        # Top-level keys must match exactly
        self.assertEqual(set(api_data.keys()), set(direct_data.keys()))
        self.assertEqual(api_data["mode"], direct_data["mode"])
        self.assertEqual(api_data["status"], direct_data["status"])
        self.assertEqual(api_data["visualization_count"], direct_data["visualization_count"])
        self.assertEqual(len(api_data["visualizations"]), len(direct_data["visualizations"]))

        # First visualization keys must match exactly
        self.assertEqual(
            set(api_data["visualizations"][0].keys()),
            set(direct_data["visualizations"][0].keys()),
        )
        self.assertEqual(
            set(api_data["visualizations"][0]["explanation"].keys()),
            set(direct_data["visualizations"][0]["explanation"].keys()),
        )


if __name__ == "__main__":
    unittest.main()
