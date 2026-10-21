from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings


class ApplicationInfoTests(unittest.TestCase):
    def test_reports_safe_runtime_metadata_without_configuration_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = ApplicationSettings(
                runtime_root=Path(temporary) / "runtime",
                run_retention_seconds=7200,
            )
            service = ApplicationRunService(settings)
            try:
                with TestClient(create_app(service=service)) as client:
                    response = client.get("/api/application-info")
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["application"]["version"], "0.1.0")
                self.assertEqual(payload["ai"]["status"], "not_configured")
                self.assertEqual(payload["data"]["retention_seconds"], 7200)
                serialized = response.text.lower()
                for forbidden in ("api_key", "password", "runtime_root", str(temporary).lower()):
                    self.assertNotIn(forbidden, serialized)
            finally:
                service.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
