const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { adaptCleaningReport, DEFAULT_CARDINALITY_CEILING } = require("../reportAdapter.js");

const projectRoot = path.resolve(__dirname, "../..");

function loadFixture(datasetId) {
  const reportPath = path.join(
    projectRoot,
    "tests",
    "pipeline1_multidomain",
    "results",
    datasetId,
    "cleaning_report.json"
  );
  return {
    report: JSON.parse(fs.readFileSync(reportPath, "utf8")),
    reportLocation: `http://localhost/${reportPath.replace(/\\/g, "/").split("/LYNX/")[1]}`
  };
}

function assertCommonMappings(report, view) {
  assert.equal(view.status, report.status);
  assert.equal(view.metrics.issuesCount, report.quality_summary.total_issues_detected);
  assert.equal(view.metrics.rows, report.validation.after_summary.row_count);
  assert.equal(view.metrics.columns, report.validation.after_summary.column_count);
  assert.equal(view.issueStatusBreakdown.reduce((sum, item) => sum + item.count, 0), view.metrics.issuesCount);
  assert.deepEqual(view.issueStatusBreakdown.map((item) => item.key), ["issue", "candidate", "observation"]);
  assert.equal(view.topIssues.length, 4);
  assert.ok(view.recommendedActions.every((item) => item.title && item.explanation));
  assert.equal(
    view.recommendedActions.length,
    report.decisions.filter((decision) => decision.decision_type === "action").length
  );
  assert.ok(view.recommendedActions.every((item) => typeof item.requiresReview === "boolean"));
  assert.ok(!Object.hasOwn(view.metrics, "overallQuality"));
  assert.ok(view.outputArtifacts.cleanedDataset.endsWith("/cleaned_dataset.csv"));
  assert.ok(!view.outputArtifacts.cleanedDataset.includes("Documents/LYNX"));
}

const surveyFixture = loadFixture("dataset_09_survey_data");
const surveyView = adaptCleaningReport(surveyFixture.report, {
  reportLocation: surveyFixture.reportLocation
});
assertCommonMappings(surveyFixture.report, surveyView);
assert.equal(surveyView.visibility.domain, false);
assert.equal(surveyView.domain, null);
assert.equal(surveyView.visibility.summary, true);
assert.equal(surveyView.summary, "Each row represents a survey response.");
assert.equal(surveyView.visibility.boundedCardinality, true);
assert.equal(surveyView.metaChips[1].label, "problems_experienced");
assert.equal(surveyView.metaChips[1].value, "27 unique values");

const businessFixture = loadFixture("dataset_13_generic_business_data");
const businessView = adaptCleaningReport(businessFixture.report, {
  reportLocation: businessFixture.reportLocation
});
assertCommonMappings(businessFixture.report, businessView);
assert.equal(businessView.visibility.domain, false);
assert.equal(businessView.visibility.summary, false);
assert.equal(businessView.summary, null);
assert.equal(businessView.visibility.boundedCardinality, true);
assert.equal(businessView.metaChips[1].label, "industry");
assert.equal(businessView.metaChips[1].value, "16 unique values");

const noBoundedView = adaptCleaningReport(surveyFixture.report, {
  reportLocation: surveyFixture.reportLocation,
  cardinalityCeiling: 1
});
assert.equal(noBoundedView.visibility.boundedCardinality, false);
assert.equal(noBoundedView.metaChips.length, 1);
assert.equal(DEFAULT_CARDINALITY_CEILING, 100);

console.log("reportAdapter integration checks passed for dataset_09 and dataset_13");
