"""Optional real-browser responsive check for the LYNX product workspace."""

from pathlib import Path
import unittest

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:  # pragma: no cover - optional local visual verification
    webdriver = None


@unittest.skipUnless(webdriver is not None, "selenium is required for browser layout checks")
class ResponsiveLayoutCheck(unittest.TestCase):
    WIDTHS = (320, 375, 430, 768, 1024, 1280, 1440)

    @classmethod
    def setUpClass(cls):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--allow-file-access-from-files")
        options.add_argument("--force-device-scale-factor=1")
        cls.driver = webdriver.Chrome(options=options)
        cls.url = (Path(__file__).resolve().parents[1] / "index.html").as_uri()

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def assert_no_page_overflow(self, width, context):
        dimensions = self.driver.execute_script(
            "return {scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth};"
        )
        self.assertLessEqual(dimensions["scroll"], dimensions["client"], f"{context} at {width}px")

    def test_required_viewports(self):
        for width in self.WIDTHS:
            with self.subTest(width=width):
                self.driver.set_window_rect(width=width, height=1100)
                self.driver.get(self.url)
                self.driver.execute_script("localStorage.setItem('lynx-theme', 'light'); location.reload();")
                WebDriverWait(self.driver, 10).until(
                    lambda browser: browser.execute_script("return Boolean(window.LYNX)")
                )
                presentation = self.driver.execute_script(
                    """
                    const header = document.querySelector('.app-header').getBoundingClientRect();
                    const heading = document.getElementById('upload-title').getBoundingClientRect();
                    const logoElement = Array.from(document.querySelectorAll('.brand-logo')).find(
                      (item) => item.getBoundingClientRect().width > 0
                    );
                    const logo = logoElement.getBoundingClientRect();
                    return {
                      headerBottom: header.bottom,
                      headingTop: heading.top,
                      headerBackground: getComputedStyle(document.querySelector('.app-header')).backgroundColor,
                      logoWidth: logo.width,
                      logoHeight: logo.height,
                      logoLoaded: logoElement.complete && logoElement.naturalWidth > 0
                    };
                    """
                )
                self.assertGreaterEqual(presentation["headingTop"], presentation["headerBottom"])
                self.assertEqual(presentation["headerBackground"], "rgb(255, 255, 255)")
                self.assertGreater(presentation["logoWidth"], 0)
                self.assertGreater(presentation["logoHeight"], 0)
                self.assertTrue(presentation["logoLoaded"])
                self.assertEqual(self.driver.find_element(By.ID, "btn-more").is_displayed(), width <= 680)
                self.assertEqual(self.driver.find_element(By.ID, "btn-settings").is_displayed(), width > 680)
                self.assertEqual(self.driver.find_element(By.ID, "btn-help").is_displayed(), width > 680)
                if width <= 680:
                    self.driver.find_element(By.ID, "btn-more").click()
                    self.assertTrue(self.driver.find_element(By.ID, "more-menu").is_displayed())
                    self.driver.find_element(By.CSS_SELECTOR, '[data-more-action="help"]').click()
                    self.assertTrue(self.driver.find_element(By.ID, "help-dialog").is_displayed())
                    self.driver.execute_script("document.getElementById('help-dialog').close()")
                self.assertFalse(self.driver.find_element(By.ID, "btn-new-dataset").is_displayed())
                self.assertFalse(self.driver.find_element(By.ID, "btn-export").is_displayed())
                self.assert_no_page_overflow(width, "upload")

                self.driver.execute_script("window.LYNX.showState('analysis')")
                analysis_presentation = self.driver.execute_script(
                    """
                    const svg = document.querySelector('.constellation');
                    const stars = svg.querySelectorAll('.active-stars .star');
                    const header = document.querySelector('.app-header').getBoundingClientRect();
                    const heading = document.getElementById('analysis-title').getBoundingClientRect();
                    return {
                      height: svg.getBoundingClientRect().height,
                      firstX: Number(stars[0].getAttribute('cx')),
                      firstY: Number(stars[0].getAttribute('cy')),
                      lastX: Number(stars[stars.length - 1].getAttribute('cx')),
                      lastY: Number(stars[stars.length - 1].getAttribute('cy')),
                      transform: getComputedStyle(svg).transform,
                      headerBottom: header.bottom,
                      headingTop: heading.top
                    };
                    """
                )
                expected_height = 102 if width <= 680 else 122 if width <= 900 else 140
                self.assertAlmostEqual(analysis_presentation["height"], expected_height, delta=1)
                self.assertLess(analysis_presentation["firstX"], analysis_presentation["lastX"])
                self.assertGreater(analysis_presentation["firstY"], analysis_presentation["lastY"])
                self.assertEqual(analysis_presentation["transform"], "none")
                self.assertGreaterEqual(analysis_presentation["headingTop"], analysis_presentation["headerBottom"])
                self.assertEqual(self.driver.find_element(By.ID, "btn-more").is_displayed(), width <= 680)
                self.assertEqual(self.driver.find_element(By.ID, "btn-settings").is_displayed(), width > 680)
                self.assertFalse(self.driver.find_element(By.ID, "btn-new-dataset").is_displayed())
                self.assertFalse(self.driver.find_element(By.ID, "btn-export").is_displayed())
                self.assert_no_page_overflow(width, "analysis")
                analysis_page_height = self.driver.execute_script(
                    "return {scroll: document.documentElement.scrollHeight, client: document.documentElement.clientHeight};"
                )
                self.assertLessEqual(analysis_page_height["scroll"], analysis_page_height["client"] + 1)

                self.driver.execute_script(
                    """
                    window.LYNX.showState('workspace');
                    const longName = 'international_customer_transaction_analysis_dataset_with_a_very_long_filename_2026.xlsx';
                    document.getElementById('workspace-dataset-name').textContent = longName;
                    document.getElementById('workspace-dataset-name').title = longName;
                    document.getElementById('workspace-dataset-meta').textContent = 'Analysis complete · 48,620 rows · 128 columns';
                    document.getElementById('overview-metrics').innerHTML = Array.from({length: 4}, (_, index) =>
                      `<div class="metric-item"><span>Metric ${index + 1}</span><strong>${index ? '128' : '48,620'}</strong></div>`
                    ).join('');
                    """
                )
                more_visible = self.driver.find_element(By.ID, "btn-more").is_displayed()
                secondary_visible = self.driver.find_element(By.ID, "btn-settings").is_displayed()
                self.assertEqual(more_visible, width <= 680)
                self.assertEqual(secondary_visible, width > 680)
                self.assertTrue(self.driver.find_element(By.ID, "btn-new-dataset").is_displayed())
                self.assertEqual(self.driver.find_element(By.ID, "btn-export").is_displayed(), width > 680)
                workspace_heading = self.driver.execute_script(
                    """
                    const header = document.querySelector('.app-header').getBoundingClientRect();
                    const heading = document.getElementById('workspace-dataset-name').getBoundingClientRect();
                    return {headerBottom: header.bottom, headingTop: heading.top};
                    """
                )
                self.assertGreaterEqual(workspace_heading["headingTop"], workspace_heading["headerBottom"])
                self.assert_no_page_overflow(width, "workspace overview")

                if width <= 680:
                    tab_scroll = self.driver.execute_script(
                        """
                        const tabs = document.querySelector('.workspace-tabs');
                        const before = tabs.scrollLeft;
                        tabs.scrollLeft = 40;
                        return {before, after: tabs.scrollLeft, scrollbarWidth: getComputedStyle(tabs).scrollbarWidth};
                        """
                    )
                    self.assertGreater(tab_scroll["after"], tab_scroll["before"])
                    self.assertEqual(tab_scroll["scrollbarWidth"], "none")

                self.driver.execute_script(
                    """
                    window.LYNX.activateTab('data-profile');
                    const list = document.getElementById('column-list');
                    list.innerHTML = Array.from({length: 120}, (_, index) => {
                      const name = `very_long_observed_column_name_${String(index + 1).padStart(3, '0')}_with_context`;
                      return `<button class="column-option" title="${name}">${name}</button>`;
                    }).join('');
                    document.getElementById('column-profile').innerHTML = '<div class="column-title-row"><div><p class="eyebrow">Selected column</p><h2>very_long_observed_column_name_001_with_context</h2></div><span class="type-tag">string</span></div>';
                    """
                )
                self.assert_no_page_overflow(width, "120-column Data Profile")

                self.driver.execute_script("window.LYNX.activateTab('prediction')")
                self.driver.execute_script("window.LYNX.showPredictionSubView('target')")
                self.assert_no_page_overflow(width, "prediction target selection")

                self.driver.execute_script("window.LYNX.showPredictionSubView('waiting')")
                self.assert_no_page_overflow(width, "prediction waiting")

                self.driver.execute_script(
                    """
                    const mockClf = {
                        target: "DonatedBloodInMarch2007",
                        task_type: "classification",
                        predictions: { row_count: 150 },
                        metrics: {
                            primary: { balanced_accuracy: 0.725, f1_macro: 0.71, roc_auc: 0.84, accuracy: 0.80 },
                            secondary: { precision_macro: 0.70, recall_macro: 0.72, mcc: 0.51, log_loss: 0.42 },
                            confusion_matrix: [[70, 10], [20, 50]]
                        }
                    };
                    const mockClfRows = {
                        offset: 0, limit: 50, total: 150,
                        rows: Array.from({length: 50}, (_, i) => ({ actual: "Yes", prediction: "Yes", confidence: 0.8499 }))
                    };
                    window.LYNX.renderPredictionResults(mockClf, mockClfRows);
                    window.LYNX.showPredictionSubView('results');
                    """
                )
                self.assert_no_page_overflow(width, "prediction classification results")

                self.driver.execute_script(
                    """
                    const mockReg = {
                        target: "MedianHouseValue",
                        task_type: "regression",
                        predictions: { row_count: 150 },
                        metrics: { r2: 0.8124, rmse: 1.25, mae: 0.80, nrmse: 0.14 }
                    };
                    const mockRegRows = {
                        offset: 0, limit: 50, total: 150,
                        rows: Array.from({length: 50}, (_, i) => ({ actual: 1.0 + i*0.1, prediction: 1.1 + i*0.1, absolute_error: 0.1 }))
                    };
                    window.LYNX.renderPredictionResults(mockReg, mockRegRows);
                    window.LYNX.showPredictionSubView('results');
                    """
                )
                self.assert_no_page_overflow(width, "prediction regression results")

                self.driver.execute_script("window.LYNX.showPredictionSubView('error')")
                self.assert_no_page_overflow(width, "prediction error")

                self.driver.execute_script("document.getElementById('help-dialog').showModal()")
                self.assert_no_page_overflow(width, "help dialog")
                self.driver.execute_script("document.getElementById('help-dialog').close()")

    def test_both_themes_at_representative_viewports(self):
        expected = {
            "light": {"body": "rgb(248, 247, 242)", "header": "rgb(255, 255, 255)"},
            "dark": {"body": "rgb(17, 16, 21)", "header": "rgb(21, 20, 25)"},
        }
        for width in (375, 768, 1440):
            for theme in ("light", "dark"):
                with self.subTest(width=width, theme=theme):
                    self.driver.set_window_rect(width=width, height=1100)
                    self.driver.get(self.url)
                    self.driver.execute_script(
                        "localStorage.setItem('lynx-theme', arguments[0]); location.reload();", theme
                    )
                    WebDriverWait(self.driver, 10).until(
                        lambda browser: browser.execute_script("return Boolean(window.LYNX)")
                    )
                    colors = self.driver.execute_script(
                        """
                        return {
                          theme: document.documentElement.dataset.theme,
                          body: getComputedStyle(document.body).backgroundColor,
                          header: getComputedStyle(document.querySelector('.app-header')).backgroundColor,
                          sidebarVisible: getComputedStyle(document.querySelector('.app-sidebar')).display !== 'none',
                          mobileBrandVisible: getComputedStyle(document.querySelector('.mobile-brand')).display !== 'none'
                        };
                        """
                    )
                    self.assertEqual(colors["theme"], theme)
                    self.assertEqual(colors["body"], expected[theme]["body"])
                    self.assertEqual(colors["header"], expected[theme]["header"])
                    self.assertEqual(colors["sidebarVisible"], width > 768)
                    self.assertEqual(colors["mobileBrandVisible"], width <= 768)
                    self.assert_no_page_overflow(width, f"{theme} upload")

        self.driver.set_window_rect(width=1440, height=1100)
        self.driver.get(self.url)
        WebDriverWait(self.driver, 10).until(
            lambda browser: browser.execute_script("return Boolean(window.LYNX)")
        )
        initial = self.driver.execute_script("return document.documentElement.dataset.theme")
        self.driver.find_element(By.ID, "btn-theme").click()
        updated = self.driver.execute_script("return document.documentElement.dataset.theme")
        self.assertNotEqual(initial, updated)
        self.driver.refresh()
        WebDriverWait(self.driver, 10).until(
            lambda browser: browser.execute_script("return Boolean(window.LYNX)")
        )
        self.assertEqual(
            self.driver.execute_script("return document.documentElement.dataset.theme"), updated
        )


if __name__ == "__main__":
    unittest.main()
