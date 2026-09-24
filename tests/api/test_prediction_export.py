"""Stage 5B: safe, read-only CSV export of public prediction rows."""

import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from api.prediction_service import PredictionJobService
from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_prediction_frontend_api import _internal_result


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runs = ApplicationRunService(ApplicationSettings(
            runtime_root=Path(self.temp.name) / "runtime", prediction_worker_count=1,
        ))
        self.jobs = PredictionJobService(self.runs, auto_execute=False)
        self.client = TestClient(create_app(service=self.runs, prediction_service=self.jobs))

    def tearDown(self):
        self.jobs.shutdown(wait=True)
        self.runs.shutdown(wait=True)
        self.temp.cleanup()

    def fixture(self, task_type="classification", rows=None, fields=None, state="complete"):
        run_id = self.runs.store.create("data.csv", ".csv")["application_run_id"]
        prediction_id = self.jobs.store.create(run_id, "outcome")["prediction_id"]
        self.jobs.store.write_result(run_id, prediction_id, _internal_result(prediction_id, task_type))
        if fields is None:
            fields = (["actual", "prediction", "probability_0", "probability_1"]
                      if task_type == "classification" else ["actual", "prediction"])
        if rows is None:
            rows = ([ ["No", "No", 0.8, 0.2], ["Yes", "No", 0.6, 0.4] ]
                    if task_type == "classification" else [[-2, -3], [4, 5]])
        self.jobs.store.write_prediction_rows(run_id, prediction_id, fields, rows)
        self.jobs.store.update(run_id, prediction_id, state=state, stage=state)
        return run_id, prediction_id

    def url(self, run_id, prediction_id, suffix="export"):
        return f"/api/runs/{run_id}/predictions/{prediction_id}/{suffix}"

    def export_rows(self, run_id, prediction_id):
        response = self.client.get(self.url(run_id, prediction_id))
        self.assertEqual(response.status_code, 200, response.text)
        return response, list(csv.DictReader(StringIO(response.text)))

    def test_classification_exact_header_parity_confidence_and_no_internal_data(self):
        run_id, prediction_id = self.fixture(rows=[
            ["No", "No", 0.8499363065, 0.1500636935],
            ["Yes", "No", "NaN", 0.4],
            ["Yes", "Yes", 1.2, 0.1],
        ])
        response, exported = self.export_rows(run_id, prediction_id)
        public = self.client.get(self.url(run_id, prediction_id, "rows")).json()["rows"]
        self.assertEqual(response.text.splitlines()[0], "actual,prediction,confidence")
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertEqual(response.headers["content-disposition"],
                         f'attachment; filename="lynx_prediction_{prediction_id}.csv"')
        self.assertEqual(len(exported), len(public))
        for csv_row, json_row in zip(exported, public):
            self.assertEqual(csv_row["actual"], json_row["actual"])
            self.assertEqual(csv_row["prediction"], json_row["prediction"])
            self.assertEqual(csv_row["confidence"],
                             "" if json_row["confidence"] is None else str(json_row["confidence"]))
        for secret in ("probability_0", "probability_1", "secret_model", "SECRET_FEATURE_A",
                       "selector", "resolver", "worker_instance_id"):
            self.assertNotIn(secret, response.text)

    def test_regression_exact_header_parity_nonfinite_and_negative_numbers(self):
        run_id, prediction_id = self.fixture("regression", rows=[
            [-2, -3], ["nan", 1], [2, "inf"], [1.7976931348623157e308, -1.7976931348623157e308],
        ])
        response, exported = self.export_rows(run_id, prediction_id)
        public = self.client.get(self.url(run_id, prediction_id, "rows")).json()["rows"]
        self.assertEqual(response.text.splitlines()[0], "actual,prediction,absolute_error")
        self.assertEqual(exported[0], {"actual": "-2.0", "prediction": "-3.0", "absolute_error": "1.0"})
        for csv_row, json_row in zip(exported, public):
            for field in ("actual", "prediction", "absolute_error"):
                self.assertEqual(csv_row[field], "" if json_row[field] is None else str(json_row[field]))
        self.assertNotIn("Infinity", response.text)
        self.assertNotIn("NaN", response.text)

    def test_csv_quoting_newlines_and_formula_neutralization(self):
        run_id, prediction_id = self.fixture(rows=[
            ["Smith, Jane", 'He said "yes"', 0.6, 0.4],
            ["line one\nline two", "=HYPERLINK(\"x\")", 0.9, 0.1],
            ["  +SUM(1,1)", "@evil", 0.5, 0.5],
        ])
        response, rows = self.export_rows(run_id, prediction_id)
        self.assertIn('"Smith, Jane"', response.text)
        self.assertIn('"He said ""yes"""', response.text)
        self.assertIn('"line one\nline two"', response.text)
        self.assertEqual(rows[1]["prediction"], "'=HYPERLINK(\"x\")")
        self.assertEqual(rows[2]["actual"], "'  +SUM(1,1)")
        self.assertEqual(rows[2]["prediction"], "'@evil")
        self.assertEqual(self.client.get(self.url(run_id, prediction_id, "rows")).json()["rows"][1]["prediction"],
                         '=HYPERLINK("x")')

    def test_lifecycle_unknown_cross_run_and_bad_id_are_controlled(self):
        for state, code in (("queued", "prediction_not_complete"),
                            ("running", "prediction_not_complete"),
                            ("failed", "prediction_failed")):
            with self.subTest(state=state):
                run_id, prediction_id = self.fixture(state=state)
                response = self.client.get(self.url(run_id, prediction_id))
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["error"]["code"], code)
        run_id, prediction_id = self.fixture(state="failed")
        self.jobs.store.update(run_id, prediction_id, error={"code": "prediction_interrupted"})
        self.assertEqual(self.client.get(self.url(run_id, prediction_id)).json()["error"]["code"], "prediction_failed")
        other_run = self.runs.store.create("other.csv", ".csv")["application_run_id"]
        self.assertEqual(self.client.get(self.url(other_run, prediction_id)).status_code, 404)
        self.assertEqual(self.client.get(self.url(run_id, str(uuid4()))).status_code, 404)
        self.assertEqual(self.client.get(self.url(run_id, "../bad")).status_code, 404)
        malformed = self.client.get(self.url(run_id, "not-a-uuid"))
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.json()["error"]["code"], "malformed_prediction_id")

    def test_missing_bad_headers_bad_rows_and_malformed_csv_are_controlled(self):
        for contents, expected in ((None, "prediction_rows_unavailable"),
                                   ("actual,prediction,probability_0,probability_0\n", "prediction_rows_invalid"),
                                   ("actual,prediction,private\n", "prediction_rows_invalid"),
                                   ("actual\n", "prediction_rows_invalid"),
                                   ("actual,prediction,probability_0\nA,B\n", "prediction_rows_invalid"),
                                   ('actual,prediction,probability_0\n"unterminated', "prediction_rows_invalid")):
            with self.subTest(contents=contents):
                run_id, prediction_id = self.fixture()
                path = self.jobs.store._paths(run_id, prediction_id)["predictions"]
                if contents is None:
                    path.unlink()
                else:
                    path.write_text(contents, encoding="utf-8")
                response = self.client.get(self.url(run_id, prediction_id))
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json()["error"]["code"], expected)
                self.assertNotIn(str(path), response.text)

    def test_large_export_is_generator_and_artifacts_are_read_only(self):
        run_id, prediction_id = self.fixture(rows=[["A", "B", 0.4, 0.6] for _ in range(2000)])
        paths = self.jobs.store._paths(run_id, prediction_id)
        originals = {name: paths[name].read_bytes() for name in ("state", "result", "predictions")}
        with patch.object(self.jobs, "rows", side_effect=AssertionError("must not paginate export")):
            chunks = self.jobs.export(run_id, prediction_id)
            self.assertNotIsInstance(chunks, (str, list, bytes))
            first = next(chunks)
            self.assertEqual(first, "actual,prediction,confidence\r\n")
            chunks.close()
        for _ in range(2):
            response, exported = self.export_rows(run_id, prediction_id)
            self.assertEqual(len(exported), 2000)
            self.assertEqual(response.text.count("\n"), 2001)
        self.assertEqual(originals, {name: paths[name].read_bytes() for name in originals})

    def test_result_and_rows_contracts_remain_available_without_ml_imports(self):
        run_id, prediction_id = self.fixture()
        with patch("api.prediction_service.importlib.import_module", side_effect=AssertionError("ML import")):
            self.assertEqual(self.client.get(self.url(run_id, prediction_id, "result")).status_code, 200)
            self.assertEqual(self.client.get(self.url(run_id, prediction_id, "rows")).status_code, 200)
            self.assertEqual(self.client.get(self.url(run_id, prediction_id)).status_code, 200)


if __name__ == "__main__":
    unittest.main()
