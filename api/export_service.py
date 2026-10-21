"""Privacy-gated exports assembled from existing verified LYNX artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import tempfile
import textwrap
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from openpyxl import Workbook

from api.run_service import ApplicationRunService, ServiceError


SECTION_ORDER = ("overview", "profiling", "prediction", "insights", "visualization")
SECTION_LABELS = {name: name.title() for name in SECTION_ORDER}
MAX_EXCEL_ROWS = 1_048_576
MAX_EXCEL_COLUMNS = 16_384
MAX_REPORT_CHARTS = 12
MAX_REPORT_TURNS = 100


class ExportArtifactStore:
    """Thread-safe snapshots of results already returned by analysis endpoints."""

    def __init__(self) -> None:
        self._automatic: dict[str, dict[str, Any]] = {}
        self._landing: dict[str, dict[str, Any]] = {}
        self._conversations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._lock = RLock()

    def set_automatic(self, dataset_id: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._automatic[dataset_id] = deepcopy(dict(payload))

    def get_automatic(self, dataset_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._automatic.get(dataset_id)
            return deepcopy(value) if value is not None else None

    def set_landing(self, dataset_id: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._landing[dataset_id] = deepcopy(dict(payload))

    def get_landing(self, dataset_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._landing.get(dataset_id)
            return deepcopy(value) if value is not None else None

    def add_conversation_result(self, dataset_id: str, conversation_id: str,
                                user_text: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            key = (dataset_id, conversation_id)
            turns = self._conversations.setdefault(key, [])
            turns.append({"user_text": user_text, "response": deepcopy(dict(payload))})
            if len(turns) > MAX_REPORT_TURNS:
                del turns[:-MAX_REPORT_TURNS]

    def get_conversation(self, dataset_id: str, conversation_id: str | None) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        with self._lock:
            return deepcopy(self._conversations.get((dataset_id, conversation_id), []))


@dataclass
class ReportSection:
    key: str
    title: str
    lines: list[str] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_filename(value: str, fallback: str = "dataset") -> str:
    stem = Path(str(value).replace("\\", "/")).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (cleaned or fallback)[:120]


def _display(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Not available"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(_display(item) for item in value)
    return str(value)


def _excel_safe(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _paragraph_xml(text: str, *, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{style_xml}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


class ExportService:
    def __init__(self, run_service: ApplicationRunService, prediction_service: Any,
                 conversation_orchestrator: Any, snapshots: ExportArtifactStore) -> None:
        self.runs = run_service
        self.predictions = prediction_service
        self.conversations = conversation_orchestrator
        self.snapshots = snapshots

    def _context(self, run_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
        run_id = self.runs.store.validate_run_id(run_id)
        status = self.runs.status(run_id)
        if status["state"] != "complete":
            raise ServiceError(409, "export_not_ready", "Exports are available after dataset preparation completes.")
        result = self.runs.pipeline1_result(run_id)
        if not result.get("privacy_clearance_passed"):
            raise ServiceError(403, "privacy_clearance_not_passed", "Exports are unavailable because privacy clearance did not pass.")
        path = self.runs.store.paths(run_id)["pipeline1"] / "cleaned_dataset.csv"
        if not path.is_file() or path.is_symlink():
            raise ServiceError(500, "cleaned_dataset_unavailable", "The cleaned dataset artifact is unavailable.")
        return status, result, path

    def dataset_csv(self, run_id: str) -> tuple[Path, str]:
        status, _result, path = self._context(run_id)
        return path, f"{_safe_filename(status['original_filename'])}_cleaned.csv"

    def dataset_xlsx(self, run_id: str) -> tuple[tempfile.SpooledTemporaryFile, str]:
        status, _result, path = self._context(run_id)
        book = Workbook(write_only=True)
        sheet = book.create_sheet("Cleaned Dataset")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source, strict=True)
                for row_index, row in enumerate(reader, start=1):
                    if row_index > MAX_EXCEL_ROWS:
                        raise ServiceError(413, "excel_row_limit", "The cleaned dataset exceeds Excel's row limit; export CSV instead.")
                    if len(row) > MAX_EXCEL_COLUMNS:
                        raise ServiceError(413, "excel_column_limit", "The cleaned dataset exceeds Excel's column limit; export CSV instead.")
                    sheet.append([_excel_safe(value) for value in row])
        except ServiceError:
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ServiceError(500, "cleaned_dataset_unavailable", "The cleaned dataset could not be exported safely.") from exc
        target = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
        book.save(target)
        target.seek(0)
        return target, f"{_safe_filename(status['original_filename'])}_cleaned.xlsx"

    def _conversation(self, run_id: str, conversation_id: str | None) -> Any | None:
        if not conversation_id:
            return None
        try:
            return self.conversations.store.get(run_id, conversation_id)
        except Exception:
            return None

    def _latest_prediction(self, run_id: str, prediction_id: str | None) -> str | None:
        if prediction_id:
            try:
                state = self.predictions.status(run_id, prediction_id)
                return prediction_id if state.get("state") == "complete" else None
            except ServiceError:
                return None
        root = self.runs.store.paths(run_id)["run"] / "predictions"
        candidates: list[tuple[str, str]] = []
        if root.is_dir() and not root.is_symlink():
            for item in root.iterdir():
                if not item.is_dir() or item.is_symlink():
                    continue
                try:
                    state = self.predictions.status(run_id, item.name)
                    if state.get("state") == "complete":
                        candidates.append((str(state.get("updated_at", "")), item.name))
                except ServiceError:
                    continue
        return max(candidates)[1] if candidates else None

    def availability(self, run_id: str, *, prediction_id: str | None = None,
                     insights_conversation_id: str | None = None,
                     visualization_conversation_id: str | None = None) -> dict[str, Any]:
        _status, result, _path = self._context(run_id)
        profile = _mapping(result.get("profiling_manifest"))
        insight_turns = self.snapshots.get_conversation(run_id, insights_conversation_id)
        visual_turns = self.snapshots.get_conversation(run_id, visualization_conversation_id)
        has_insight_turn = any(_mapping(turn.get("response")).get("intent") == "INSIGHT" and _mapping(turn.get("response")).get("status") == "READY" for turn in insight_turns)
        has_visual_turn = any(_mapping(turn.get("response")).get("intent") == "VISUALIZATION" and _mapping(turn.get("response")).get("status") == "READY" for turn in visual_turns)
        selected_prediction = self._latest_prediction(run_id, prediction_id)
        available = {
            "overview": True,
            "profiling": profile.get("profile_status") in {"completed", "partial"},
            "prediction": selected_prediction is not None,
            "insights": self.snapshots.get_landing(run_id) is not None or has_insight_turn,
            "visualization": self.snapshots.get_automatic(run_id) is not None or has_visual_turn,
        }
        return {
            "run_id": run_id,
            "ready": True,
            "prediction_id": selected_prediction,
            "sections": [
                {"key": key, "label": SECTION_LABELS[key], "status": "available" if available[key] else "unavailable"}
                for key in SECTION_ORDER
            ],
        }

    def _overview(self, result: Mapping[str, Any]) -> ReportSection:
        report = _mapping(result.get("pipeline1_report"))
        validation = _mapping(report.get("validation"))
        summary = _mapping(validation.get("after_summary"))
        eligibility = _mapping(result.get("prediction_eligibility"))
        lines = [
            f"Prepared rows: {_display(summary.get('row_count', _mapping(result.get('prepared_dataset')).get('rows')))}",
            f"Columns: {_display(summary.get('column_count', _mapping(result.get('prepared_dataset')).get('columns')))}",
            f"Missing cells: {_display(summary.get('total_missing_cells'))}",
            f"Duplicate rows: {_display(summary.get('duplicate_row_count'))}",
            f"Validation: {'Passed' if validation.get('passed') else 'Not passed'}",
            f"Privacy clearance: {'Passed' if result.get('privacy_clearance_passed') else 'Not passed'}",
            f"Prediction eligible: {'Yes' if eligibility.get('eligible') else 'No'}",
        ]
        return ReportSection("overview", "Overview", lines)

    def _profiling(self, result: Mapping[str, Any]) -> ReportSection:
        manifest = _mapping(result.get("profiling_manifest"))
        lines = [
            f"Profile status: {_display(manifest.get('profile_status'))}",
            f"Columns profiled: {_display(manifest.get('total_columns'))}",
            f"Semantic roles detected: {_display(manifest.get('semantic_roles_detected'))}",
            f"Columns requiring cleaning: {_display(manifest.get('columns_requiring_cleaning'))}",
        ]
        for column in list(manifest.get("columns") or [])[:100]:
            if not isinstance(column, Mapping):
                continue
            lines.append(
                f"{column.get('column_name')}: type={_display(column.get('physical_dtype'))}; "
                f"role={_display(column.get('semantic_role') or column.get('semantic_type'))}; "
                f"missing={_display(column.get('missing_percentage'))}%; unique={_display(column.get('unique_count'))}; "
                f"decision={_display(column.get('decision_explanation') or _mapping(column.get('primary_action')).get('rationale'))}"
            )
        return ReportSection("profiling", "Profiling", lines)

    def _prediction(self, run_id: str, prediction_id: str) -> ReportSection:
        result = self.predictions.result(run_id, prediction_id)
        rows = self.predictions.rows(run_id, prediction_id, offset=0, limit=25)
        lines = [
            f"Target: {_display(result.get('target'))}",
            f"Task type: {_display(result.get('task_type'))}",
            f"Completed at: {_display(result.get('completed_at'))}",
            f"Evaluated rows: {_display(_mapping(result.get('dataset')).get('test_rows'))}",
        ]
        metrics = _mapping(result.get("metrics"))
        primary = _mapping(metrics.get("primary")) if "primary" in metrics else metrics
        for key, value in primary.items():
            if isinstance(value, (str, int, float)) or value is None:
                lines.append(f"{key.replace('_', ' ').title()}: {_display(value)}")
        importance = _mapping(result.get("feature_importance"))
        charts: list[dict[str, Any]] = []
        if importance.get("available"):
            lines.append(
                f"Feature importance method: {_display(importance.get('method'))}; metric: {_display(importance.get('metric'))}; "
                f"evaluation scope: {_display(importance.get('evaluation_scope'))}"
            )
            for feature in list(importance.get("features") or [])[:20]:
                if isinstance(feature, Mapping):
                    lines.append(f"Feature {_display(feature.get('feature'))}: {_display(feature.get('importance'))}")
            feature_rows = [dict(feature) for feature in list(importance.get("features") or [])[:20] if isinstance(feature, Mapping)]
            if feature_rows:
                charts.append({"chart_type": "BAR", "chart_spec": {
                    "family": "BAR", "title": "Feature Importance",
                    "data_ref": {"rows": feature_rows},
                    "encodings": [
                        {"channel": "X", "field": "feature", "semantic_label": "Feature"},
                        {"channel": "Y", "field": "importance", "semantic_label": "Importance"},
                    ],
                }})
        task_type = result.get("task_type")
        if task_type == "classification":
            matrix = metrics.get("confusion_matrix")
            labels = result.get("class_labels")
            if isinstance(matrix, list) and isinstance(labels, list) and len(matrix) == len(labels):
                matrix_rows = [
                    {"actual": str(actual), "predicted": str(predicted), "count": value}
                    for actual_index, actual in enumerate(labels)
                    for predicted_index, predicted in enumerate(labels)
                    for value in [matrix[actual_index][predicted_index]]
                ]
                charts.append({"chart_type": "HEATMAP", "chart_spec": {
                    "family": "HEATMAP", "title": "Confusion Matrix",
                    "data_ref": {"rows": matrix_rows},
                    "encodings": [
                        {"channel": "X", "field": "predicted", "semantic_label": "Predicted"},
                        {"channel": "Y", "field": "actual", "semantic_label": "Actual"},
                        {"channel": "COLOR", "field": "count", "semantic_label": "Count"},
                    ],
                }})
        elif task_type == "regression":
            points = [row for row in rows.get("rows", []) if isinstance(row, Mapping) and row.get("actual") is not None and row.get("prediction") is not None]
            if points:
                charts.append({"chart_type": "SCATTER", "chart_spec": {
                    "family": "SCATTER", "title": "Actual vs Predicted",
                    "data_ref": {"rows": points},
                    "encodings": [
                        {"channel": "X", "field": "actual", "semantic_label": "Actual"},
                        {"channel": "Y", "field": "prediction", "semantic_label": "Predicted"},
                    ],
                }})
        lines.append("Prediction rows (bounded preview):")
        for row in list(rows.get("rows") or [])[:25]:
            lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return ReportSection("prediction", "Prediction", lines, charts)

    def _insights(self, run_id: str, conversation_id: str | None) -> ReportSection:
        lines: list[str] = []
        landing = self.snapshots.get_landing(run_id)
        if landing:
            lines.append("Dataset summary: " + _display(landing.get("summary_text")))
        for turn in self.snapshots.get_conversation(run_id, conversation_id):
            response = _mapping(turn.get("response"))
            payload = _mapping(response.get("insight"))
            if response.get("intent") != "INSIGHT" or response.get("status") != "READY":
                continue
            lines.append("Question: " + _display(turn.get("user_text")))
            lines.append("Answer: " + _display(payload.get("answer") or response.get("message") or payload.get("message")))
        return ReportSection("insights", "Insights", lines or ["No completed insight interaction was available."])

    def _visualization(self, run_id: str, conversation_id: str | None) -> ReportSection:
        charts: list[dict[str, Any]] = []
        payload = self.snapshots.get_automatic(run_id)
        if payload:
            charts.extend(item for item in payload.get("visualizations", []) if isinstance(item, dict))
        for turn in self.snapshots.get_conversation(run_id, conversation_id):
            response = _mapping(turn.get("response"))
            payload = _mapping(response.get("visualization"))
            if response.get("intent") != "VISUALIZATION" or response.get("status") != "READY":
                continue
            charts.extend(item for item in payload.get("visualizations", []) if isinstance(item, dict))
        charts = charts[:MAX_REPORT_CHARTS]
        lines: list[str] = []
        for chart in charts:
            spec = _mapping(chart.get("chart_spec"))
            explanation = _mapping(chart.get("xai")) or _mapping(chart.get("explanation"))
            lines.append(_display(spec.get("title") or chart.get("title") or chart.get("chart_type")))
            lines.append(_display(explanation.get("what_it_shows") or explanation.get("why_this_chart")))
            if str(spec.get("family", "")).upper() not in {"BAR", "LINE", "AREA", "PIE", "DOUGHNUT", "SCATTER", "HISTOGRAM", "HEATMAP"}:
                lines.append("Chart image not embedded: this verified chart family requires the interactive renderer; its explanation is preserved.")
        return ReportSection("visualization", "Visualization", lines or ["No completed visualization was available."], charts)

    def _sections(self, run_id: str, selected: Sequence[str], *, prediction_id: str | None,
                  insights_conversation_id: str | None, visualization_conversation_id: str | None) -> tuple[str, list[ReportSection]]:
        status, result, _path = self._context(run_id)
        availability = self.availability(
            run_id, prediction_id=prediction_id,
            insights_conversation_id=insights_conversation_id,
            visualization_conversation_id=visualization_conversation_id,
        )
        status_by_key = {item["key"]: item["status"] for item in availability["sections"]}
        normalized = list(dict.fromkeys(str(item).lower() for item in selected))
        if not normalized or any(item not in SECTION_ORDER for item in normalized):
            raise ServiceError(422, "invalid_report_sections", "Choose at least one valid report section.")
        unavailable = [item for item in normalized if status_by_key[item] != "available"]
        if unavailable:
            raise ServiceError(409, "report_section_unavailable", f"Selected report sections are unavailable: {', '.join(unavailable)}.")
        chosen_prediction = availability.get("prediction_id")
        builders = {
            "overview": lambda: self._overview(result),
            "profiling": lambda: self._profiling(result),
            "prediction": lambda: self._prediction(run_id, chosen_prediction),
            "insights": lambda: self._insights(run_id, insights_conversation_id),
            "visualization": lambda: self._visualization(run_id, visualization_conversation_id),
        }
        return _safe_filename(status["original_filename"]), [builders[key]() for key in SECTION_ORDER if key in normalized]

    @staticmethod
    def _chart_png(item: Mapping[str, Any]) -> bytes | None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            spec = _mapping(item.get("chart_spec"))
            data = _mapping(spec.get("data_ref"))
            rows = list(data.get("rows") or [])[:100]
            encodings = list(spec.get("encodings") or [])
            by_channel = {str(e.get("channel", "")).upper(): e for e in encodings if isinstance(e, Mapping)}
            x = by_channel.get("X") or by_channel.get("CATEGORY")
            y = by_channel.get("Y") or by_channel.get("VALUE")
            family = str(spec.get("family") or item.get("chart_type") or "BAR").upper()
            if not rows or not x or not y:
                return None
            xv = [row.get(x.get("field")) for row in rows]
            yv = [row.get(y.get("field")) for row in rows]
            fig, ax = plt.subplots(figsize=(8.2, 3.7), dpi=150)
            colors = ["#4B179B", "#7248C8", "#9676DB", "#B49CE7", "#D1C4F2"]
            if family in {"LINE", "AREA"}:
                ax.plot(xv, yv, color=colors[0], marker="o", linewidth=2)
                if family == "AREA": ax.fill_between(range(len(yv)), yv, color=colors[2], alpha=.22)
            elif family in {"PIE", "DOUGHNUT"}:
                ax.pie(yv, labels=xv, colors=[colors[i % len(colors)] for i in range(len(yv))], wedgeprops={"width": .55} if family == "DOUGHNUT" else None)
            elif family == "SCATTER":
                ax.scatter(xv, yv, color=colors[0], alpha=.75)
            elif family == "HISTOGRAM" and all("count" in row for row in rows):
                ax.bar(range(len(rows)), [row["count"] for row in rows], color=colors[1])
            elif family == "HEATMAP":
                color = by_channel.get("COLOR") or by_channel.get("VALUE")
                if not color:
                    return None
                xs = list(dict.fromkeys(xv)); ys = list(dict.fromkeys(yv))
                matrix = [[0 for _ in xs] for _ in ys]
                for row in rows:
                    matrix[ys.index(row.get(y.get("field")))][xs.index(row.get(x.get("field")))] = row.get(color.get("field"), 0)
                image = ax.imshow(matrix, cmap="Purples", aspect="auto")
                ax.set_xticks(range(len(xs)), [str(value) for value in xs], rotation=25, ha="right")
                ax.set_yticks(range(len(ys)), [str(value) for value in ys])
                fig.colorbar(image, ax=ax, fraction=.03, pad=.03)
            else:
                if family != "BAR":
                    return None
                ax.bar(range(len(yv)), yv, color=[colors[i % len(colors)] for i in range(len(yv))])
                ax.set_xticks(range(len(xv)), [str(value) for value in xv], rotation=25, ha="right")
            ax.set_title(str(spec.get("title") or item.get("chart_type") or "Verified visualization"), loc="left", color="#17123B", weight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#EAE7F2", linewidth=.7)
            fig.tight_layout()
            output = io.BytesIO()
            fig.savefig(output, format="png", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return output.getvalue()
        except Exception:
            return None

    def report(self, run_id: str, *, format_name: str, sections: Sequence[str],
               prediction_id: str | None = None, insights_conversation_id: str | None = None,
               visualization_conversation_id: str | None = None) -> tuple[bytes, str, str]:
        if format_name not in {"pdf", "docx"}:
            raise ServiceError(422, "invalid_report_format", "Report format must be PDF or DOCX.")
        name, content = self._sections(
            run_id, sections, prediction_id=prediction_id,
            insights_conversation_id=insights_conversation_id,
            visualization_conversation_id=visualization_conversation_id,
        )
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        title = f"LYNX Analysis Report — {name}"
        if format_name == "pdf":
            data = self._pdf(title, generated, content)
            return data, f"{name}_analysis_report.pdf", "application/pdf"
        data = self._docx(title, generated, content)
        return data, f"{name}_analysis_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    @staticmethod
    def _pdf(title: str, generated: str, sections: Sequence[ReportSection]) -> bytes:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        output = io.BytesIO()
        with PdfPages(output, metadata={"Title": title, "Author": "LYNX", "Subject": "Verified analysis export"}) as pdf:
            def page(heading: str, lines: Sequence[str], image: bytes | None = None) -> None:
                fig = plt.figure(figsize=(8.27, 11.69), dpi=120, facecolor="#FBFAFD")
                fig.text(.08, .94, "LYNX", color="#4B179B", fontsize=13, weight="bold")
                fig.text(.08, .90, heading, color="#17123B", fontsize=18, weight="bold")
                fig.text(.08, .872, generated, color="#706A84", fontsize=8)
                y = .83
                for raw in lines:
                    for wrapped in textwrap.wrap(str(raw), width=100) or [""]:
                        fig.text(.08, y, wrapped, color="#37324C", fontsize=8.5)
                        y -= .022
                        if y < .08: break
                    if y < .08: break
                if image:
                    try:
                        array = plt.imread(io.BytesIO(image), format="png")
                        axis = fig.add_axes([.08, .08, .84, .35])
                        axis.imshow(array); axis.axis("off")
                    except Exception: pass
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            page(title, ["Generated from privacy-cleared, verified LYNX artifacts."])
            for section in sections:
                if section.charts:
                    chunks = [section.lines[i:i + 32] for i in range(0, len(section.lines), 32)] or [[]]
                    for index, chunk in enumerate(chunks):
                        page(section.title + (f" (continued {index + 1})" if index else ""), chunk)
                    for chart in section.charts:
                        image = ExportService._chart_png(chart)
                        spec = _mapping(chart.get("chart_spec"))
                        page(str(spec.get("title") or section.title), [
                            _display((_mapping(chart.get("xai")) or _mapping(chart.get("explanation"))).get("what_it_shows"))
                        ], image)
                else:
                    chunks = [section.lines[i:i + 32] for i in range(0, len(section.lines), 32)] or [[]]
                    for index, chunk in enumerate(chunks):
                        page(section.title + (f" (continued {index + 1})" if index else ""), chunk)
        return output.getvalue()

    @staticmethod
    def _docx(title: str, generated: str, sections: Sequence[ReportSection]) -> bytes:
        paragraphs = [_paragraph_xml("LYNX", style="Title"), _paragraph_xml(title, style="Heading1"), _paragraph_xml(generated)]
        images: list[bytes] = []
        relationships: list[str] = []
        for section in sections:
            paragraphs.append(_paragraph_xml(section.title, style="Heading1"))
            paragraphs.extend(_paragraph_xml(line) for line in section.lines)
            for chart in section.charts:
                payload = ExportService._chart_png(chart)
                if not payload: continue
                images.append(payload); rid = f"rId{len(images)}"
                relationships.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{len(images)}.png"/>')
                paragraphs.append(
                    f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="5486400" cy="2468880"/><wp:docPr id="{len(images)}" name="Chart {len(images)}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="image{len(images)}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="5486400" cy="2468880"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
                )
        document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>' + ''.join(paragraphs) + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>'
        styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:color w:val="37324C"/><w:sz w:val="20"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:color w:val="4B179B"/><w:b/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:color w:val="17123B"/><w:b/><w:sz w:val="30"/></w:rPr></w:style></w:styles>'
        types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
        root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
        doc_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(relationships) + '</Relationships>'
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as package:
            package.writestr("[Content_Types].xml", types); package.writestr("_rels/.rels", root_rels)
            package.writestr("word/document.xml", document); package.writestr("word/styles.xml", styles)
            package.writestr("word/_rels/document.xml.rels", doc_rels)
            for index, payload in enumerate(images, start=1): package.writestr(f"word/media/image{index}.png", payload)
        return output.getvalue()


__all__ = ["ExportArtifactStore", "ExportService", "SECTION_ORDER"]
