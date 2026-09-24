"""End-to-end real integration check for LYNX Stage 5A.1 against the real FastAPI backend."""

import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import unittest
import urllib.request

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    webdriver = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "datasets" / "valid" / "Sales_Data.xlsx"
PYTHON_EXE = Path(sys.executable)
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
LOG_FILE = PROJECT_ROOT / "uvicorn_integration.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@unittest.skipUnless(webdriver is not None, "selenium is required for end-to-end integration check")
class RealPredictionIntegrationCheck(unittest.TestCase):
    server_process = None
    log_handle = None

    @classmethod
    def setUpClass(cls):
        # Open log file for server output to avoid pipe-buffer blocking on Windows
        cls.log_handle = open(LOG_FILE, "w", encoding="utf-8")
        server_cmd = [
            str(PYTHON_EXE),
            "-m", "uvicorn",
            "api.server:app",
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--log-level", "info",
        ]
        logger.info("Starting real FastAPI server: %s", " ".join(server_cmd))
        cls.server_process = subprocess.Popen(
            server_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=cls.log_handle,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to become responsive
        deadline = time.time() + 15
        started = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{BASE_URL}/", timeout=1) as response:
                    if response.status == 200:
                        started = True
                        break
            except Exception:
                time.sleep(0.5)

        if not started:
            cls.tearDownClass()
            raise RuntimeError("FastAPI server failed to start within timeout.")

        logger.info("FastAPI server successfully listening at %s", BASE_URL)

        # Initialize headless Chrome with logging
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--force-device-scale-factor=1")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})
        cls.driver = webdriver.Chrome(options=options)
        cls.driver.set_window_rect(width=1280, height=900)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "driver") and cls.driver:
            cls.driver.quit()
        if cls.server_process:
            logger.info("Terminating FastAPI server process...")
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
        if cls.log_handle:
            cls.log_handle.close()

    def test_real_dataset_prediction_end_to_end(self):
        driver = self.driver
        driver.get(f"{BASE_URL}/")

        # 1. Upload Screen Verification
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "dataset-file-input"))
        )
        self.assertTrue(driver.find_element(By.ID, "view-upload").is_displayed())
        logger.info("A. Upload screen verified.")

        # 2. Upload real dataset file (Sales_Data.xlsx)
        file_input = driver.find_element(By.ID, "dataset-file-input")
        file_input.send_keys(str(DATASET_PATH))

        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "selected-file-card"))
        )
        self.assertEqual(driver.find_element(By.ID, "selected-file-name").text, "Sales_Data.xlsx")
        analyze_btn = driver.find_element(By.ID, "btn-run-analysis")
        self.assertTrue(analyze_btn.is_enabled())

        # Click analyze
        analyze_btn.click()
        logger.info("B. Analysis requested for Sales_Data.xlsx.")

        # 3. Verify Analysis waiting state
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "view-analysis"))
        )
        constellation = driver.find_element(By.CSS_SELECTOR, "#view-analysis .constellation")
        self.assertTrue(constellation.is_displayed())
        self.assertEqual(len(constellation.find_elements(By.CSS_SELECTOR, ".star")), 8)
        self.assertTrue(driver.find_element(By.ID, "analysis-elapsed").is_displayed())
        logger.info("G. Analysis constellation loader and timer verified.")

        # 4. Wait for Pipeline 1 completion and workspace open
        logger.info("Waiting for real Pipeline 1 to complete (takes ~60-70s)...")
        WebDriverWait(driver, 180).until(
            EC.visibility_of_element_located((By.ID, "view-workspace"))
        )
        logger.info("B. Pipeline 1 completed and workspace opened successfully.")

        # Verify Overview tab contains dataset info
        workspace_name = driver.find_element(By.ID, "workspace-dataset-name").text
        self.assertTrue(bool(workspace_name))
        logger.info("Workspace dataset name displayed: %s", workspace_name)

        # 5. Switch to Prediction Tab
        pred_tab_btn = driver.find_element(By.ID, "tab-prediction")
        pred_tab_btn.click()
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "prediction-target-section"))
        )
        logger.info("C. Switched to Prediction tab.")

        # 6. Verify real target columns populated
        target_select = driver.find_element(By.ID, "prediction-target")
        options = [opt.get_attribute("value") for opt in target_select.find_elements(By.TAG_NAME, "option") if opt.get_attribute("value")]
        logger.info("Target columns detected: %s", options)
        self.assertIn("Category", options)
        self.assertIn("Profit", options)
        self.assertIn("Sales", options)

        # 7. Run Prediction button is disabled before selection
        run_btn = driver.find_element(By.ID, "btn-run-prediction")
        self.assertFalse(run_btn.is_enabled())
        logger.info("D. Run Prediction button disabled without selection.")

        # 8. Select valid classification target: "Category"
        driver.execute_script("""
            const select = document.getElementById('prediction-target');
            select.value = 'Category';
            select.dispatchEvent(new Event('change'));
        """)
        self.assertTrue(run_btn.is_enabled())
        logger.info("D. Run Prediction button enabled after selecting 'Category'.")

        # 9. Click Run Prediction -> verify job creation & waiting state
        run_btn.click()
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "prediction-waiting-section"))
        )
        pred_constellation = driver.find_element(By.CSS_SELECTOR, "#prediction-waiting-section .constellation")
        self.assertTrue(pred_constellation.is_displayed())
        self.assertEqual(len(pred_constellation.find_elements(By.CSS_SELECTOR, ".star")), 8)
        self.assertTrue(driver.find_element(By.ID, "prediction-elapsed").is_displayed())
        self.assertEqual(driver.find_element(By.ID, "prediction-waiting-target-name").text, "Category")

        # Check that internal backend stages do NOT appear
        waiting_text = driver.find_element(By.ID, "prediction-waiting-section").text
        for forbidden in ["task_resolution", "model_selection", "evaluation", "training"]:
            self.assertNotIn(forbidden, waiting_text)
        logger.info("E, F, G, H. Prediction UUID created, waiting section active, Lynx constellation displayed, no internal stages.")

        # 10. Wait for Prediction completion & results rendering
        logger.info("Waiting for real prediction execution (frozen selector + models)...")
        WebDriverWait(driver, 120).until(
            EC.visibility_of_element_located((By.ID, "prediction-results-section"))
        )
        logger.info("I, J. Prediction reached complete; safe result rendered!")

        # 11. Verify Safe Results Content
        target_display = driver.find_element(By.ID, "pred-summary-target").text
        task_display = driver.find_element(By.ID, "pred-summary-task").text
        rows_display = driver.find_element(By.ID, "pred-summary-rows").text
        self.assertEqual(target_display, "Category")
        self.assertEqual(task_display, "Classification")
        logger.info("Summary: Target=%s, Task=%s, Evaluated rows=%s", target_display, task_display, rows_display)

        # Primary metrics
        primary_cards = driver.find_elements(By.CSS_SELECTOR, "#prediction-primary-metrics .metric-card")
        self.assertEqual(len(primary_cards), 4)
        card_texts = [card.text for card in primary_cards]
        logger.info("Primary metric cards: %s", card_texts)
        self.assertTrue(any("Balanced Accuracy" in t for t in card_texts))
        self.assertTrue(any("F1 Score" in t for t in card_texts))
        self.assertTrue(any("ROC AUC" in t for t in card_texts))
        self.assertTrue(any("Accuracy" in t for t in card_texts))

        # Confusion Matrix
        cm_table = driver.find_element(By.CSS_SELECTOR, ".confusion-matrix-table")
        self.assertTrue(cm_table.is_displayed())
        cm_text = cm_table.text
        self.assertNotIn("%", cm_text, "Confusion matrix must render counts only and no percentages")
        logger.info("Confusion matrix table verified: raw counts only.")

        # 12. Prediction Rows Table and Pagination
        table = driver.find_element(By.ID, "prediction-rows-table")
        self.assertTrue(table.is_displayed())
        headers = [th.text for th in table.find_elements(By.CSS_SELECTOR, "thead th")]
        self.assertEqual(headers, ["#", "Actual", "Prediction", "Confidence"])

        rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
        self.assertEqual(len(rows), 20)  # test split has 20 rows for Sales_Data (80 train, 20 test)
        first_row_cells = [td.text for td in rows[0].find_elements(By.TAG_NAME, "td")]
        self.assertEqual(first_row_cells[0], "1", "First row sequence index # must be 1")
        self.assertNotEqual(first_row_cells[1], "")
        self.assertNotEqual(first_row_cells[2], "")
        self.assertIn("%", first_row_cells[3], "Confidence should be formatted as % or —")
        logger.info("K. Prediction rows verified (first row: %s)", first_row_cells)

        # Pagination info
        page_info = driver.find_element(By.ID, "pred-page-info").text
        self.assertIn("1–20 of 20 rows", page_info)
        prev_btn = driver.find_element(By.ID, "btn-pred-prev-page")
        next_btn = driver.find_element(By.ID, "btn-pred-next-page")
        self.assertFalse(prev_btn.is_enabled())
        self.assertFalse(next_btn.is_enabled())
        logger.info("L. Pagination bounds verified: %s", page_info)

        # 13. Verify ABSENCE of model names, rankings, and selector internals
        page_content = driver.find_element(By.ID, "panel-prediction").text
        for forbidden in [
            "LightGBM", "CatBoost", "XGBoost", "RandomForest", "TabPFN",
            "selected_model", "candidate_ranking", "model_rank", "selector_score",
            "probability_0", "probability_1",
        ]:
            self.assertNotIn(forbidden, page_content, f"Forbidden selector/internal text found: {forbidden}")
        logger.info("M. Confirmed zero internal model/ranking/selector disclosures in UI.")

        # 14. Verify "Run another prediction" and run real Regression on target ("Sales")
        run_another_btn = driver.find_element(By.ID, "btn-prediction-run-another")
        run_another_btn.click()
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "prediction-target-section"))
        )
        self.assertTrue(driver.find_element(By.ID, "prediction-target-section").is_displayed())
        logger.info("Run another prediction returned to target selection cleanly.")

        # Select valid regression target: "Sales"
        driver.execute_script("""
            const select = document.getElementById('prediction-target');
            select.value = 'Sales';
            select.dispatchEvent(new Event('change'));
        """)
        run_btn = driver.find_element(By.ID, "btn-run-prediction")
        self.assertTrue(run_btn.is_enabled())
        run_btn.click()

        # Wait for regression results
        logger.info("Waiting for real regression execution on 'Sales'...")
        WebDriverWait(driver, 120).until(
            EC.visibility_of_element_located((By.ID, "prediction-results-section"))
        )
        self.assertEqual(driver.find_element(By.ID, "pred-summary-target").text, "Sales")
        self.assertEqual(driver.find_element(By.ID, "pred-summary-task").text, "Regression")
        logger.info("Regression result verified: target=Sales, task=Regression")

        # Verify 4 regression metric cards: R², RMSE, MAE, NRMSE
        reg_cards = [card.text for card in driver.find_elements(By.CSS_SELECTOR, "#prediction-primary-metrics .metric-card")]
        logger.info("Regression metric cards: %s", reg_cards)
        self.assertTrue(any("R²" in t for t in reg_cards))
        self.assertTrue(any("RMSE" in t for t in reg_cards))
        self.assertTrue(any("MAE" in t for t in reg_cards))
        self.assertTrue(any("NRMSE" in t for t in reg_cards))

        # Verify Actual vs Predicted scatter plot SVG
        scatter_svg = driver.find_element(By.CSS_SELECTOR, ".scatter-plot-svg")
        self.assertTrue(scatter_svg.is_displayed())
        logger.info("Regression scatter plot SVG verified.")

        # Verify Regression prediction rows table: Absolute Error column
        reg_table = driver.find_element(By.ID, "prediction-rows-table")
        reg_headers = [th.text for th in reg_table.find_elements(By.CSS_SELECTOR, "thead th")]
        self.assertEqual(reg_headers, ["#", "Actual", "Prediction", "Absolute Error"])
        first_reg_row = [td.text for td in reg_table.find_elements(By.CSS_SELECTOR, "tbody tr")[0].find_elements(By.TAG_NAME, "td")]
        self.assertEqual(first_reg_row[0], "1")
        self.assertNotEqual(first_reg_row[3], "")  # formatted absolute error
        logger.info("Regression prediction rows verified: Absolute Error present (row: %s)", first_reg_row)

        # 15. Verify "Run another prediction" and Controlled Error on Unresolved Target ("Order Date")
        driver.find_element(By.ID, "btn-prediction-run-another").click()
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "prediction-target-section"))
        )

        driver.execute_script("""
            const select = document.getElementById('prediction-target');
            select.value = 'Order Date';
            select.dispatchEvent(new Event('change'));
        """)
        run_btn = driver.find_element(By.ID, "btn-run-prediction")
        self.assertTrue(run_btn.is_enabled())
        run_btn.click()

        # Wait for failure view
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.ID, "prediction-error-section"))
        )
        error_text = driver.find_element(By.ID, "prediction-error-text").text
        logger.info("Controlled error message observed for unresolved target: %s", error_text)
        self.assertEqual(error_text, "LYNX could not determine a reliable prediction type for this target.")
        for forbidden in ["Traceback", "Exception", "File \"", "task_type_resolver", "<html>"]:
            self.assertNotIn(forbidden, error_text)

        # Click "Try again"
        retry_btn = driver.find_element(By.ID, "btn-prediction-retry")
        retry_btn.click()
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "prediction-target-section"))
        )
        self.assertTrue(driver.find_element(By.ID, "prediction-target-section").is_displayed())
        logger.info("Controlled error recovery via 'Try again' verified.")

        # 16. Inspect Server Log to report HTTP lifecycle
        logger.info("HTTP Server Logs summary:")
        cls = self.__class__
        cls.log_handle.flush()
        if LOG_FILE.is_file():
            content = LOG_FILE.read_text(encoding="utf-8", errors="replace")
            http_lines = [line.strip() for line in content.splitlines() if "HTTP/1.1" in line]
            for line in http_lines[:25]:
                logger.info("  %s", line)


if __name__ == "__main__":
    unittest.main()

