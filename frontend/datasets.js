// Real Pipeline 1 report sources used by the static dashboard.
// Serve the repository root over HTTP so these report URLs remain accessible.

const LYNX_REPORT_SOURCES = {
  "survey_data": {
    "label": "Survey Data",
    "reportUrl": "../tests/pipeline1_multidomain/results/dataset_09_survey_data/cleaning_report.json"
  },
  "generic_business_data": {
    "label": "Generic Business Data",
    "reportUrl": "../tests/pipeline1_multidomain/results/dataset_13_generic_business_data/cleaning_report.json"
  }
};
