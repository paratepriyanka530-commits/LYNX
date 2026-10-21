"""Allow-listed pandas backend for canonical AnalyticalPlan execution."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from pipelines.pipeline3_results.execution.base import AnalyticalExecutor
from pipelines.pipeline3_results.execution.schemas import (
    ExecutionCoverage,
    ExecutionError,
    ExecutionMetadata,
    ResultField,
    ResultSchema,
    ResultVerification,
    VerifiedResult,
)
from pipelines.pipeline3_results.execution.verifier import verify_result
from pipelines.pipeline3_results.grounding.engine import (
    is_category_eligible,
    is_numeric_eligible,
)
from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetKnowledge,
)
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan, PlanFilter
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _result_id(plan_id: str, payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{plan_id}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"result_{digest}"


def _metadata(plan: AnalyticalPlan, *, details: dict[str, Any] | None = None) -> ExecutionMetadata:
    return ExecutionMetadata(
        backend="pandas",
        operation=plan.operation.value,
        expressions=[item.model_dump(mode="json") for item in plan.measures],
        groupings=[item.model_dump(mode="json") for item in [*plan.dimensions, *plan.temporal]],
        filters_applied=[item.model_dump(mode="json") for item in plan.filters],
        sort=[item.model_dump(mode="json") for item in plan.sort],
        temporal_bucket=(plan.temporal[0].model_dump(mode="json") if plan.temporal and plan.temporal[0].kind == "TIME_BUCKET" else None),
        details=details or {},
    )


def _empty_coverage(rows: int) -> ExecutionCoverage:
    return ExecutionCoverage(source_rows=rows, filtered_rows=rows, usable_rows=0)


def _failure(
    plan: AnalyticalPlan,
    rows: int,
    status: str,
    code: str,
    message: str,
    *,
    result_type: str = "TABLE",
    coverage: ExecutionCoverage | None = None,
) -> VerifiedResult:
    payload = {"status": status, "code": code, "message": message}
    return VerifiedResult(
        result_id=_result_id(plan.plan_id, payload),
        plan_id=plan.plan_id,
        status=status,  # type: ignore[arg-type]
        result_type=result_type,  # type: ignore[arg-type]
        schema=ResultSchema(),
        coverage=coverage or _empty_coverage(rows),
        execution_metadata=_metadata(plan),
        errors=[ExecutionError(code=code, message=message)],  # type: ignore[arg-type]
        verification=ResultVerification(status="NOT_RUN"),
    )


def _numeric(series: pd.Series) -> tuple[pd.Series, pd.Series, int, int]:
    converted = pd.to_numeric(series, errors="coerce")
    nulls = int(series.isna().sum())
    finite = converted.notna() & np.isfinite(converted.astype(float))
    invalid = int((series.notna() & ~finite).sum())
    return converted, finite, nulls, invalid


def _aggregate(series: pd.Series, operation: str) -> float:
    functions = {
        "SUM": series.sum,
        "MEAN": series.mean,
        "MEDIAN": series.median,
        "MIN": series.min,
        "MAX": series.max,
    }
    return float(functions[operation]())


def _stable_key(value: Any) -> tuple[str, str]:
    safe = _json_scalar(value)
    return type(safe).__name__, str(safe).casefold()


class PandasAnalyticalExecutor(AnalyticalExecutor):
    """Fixed pandas operations only; no eval, query strings, or generated code."""

    def execute(
        self,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
        dataset_knowledge: DatasetKnowledge,
    ) -> VerifiedResult:
        if not isinstance(plan, AnalyticalPlan):
            raise TypeError("plan must be an AnalyticalPlan")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if not isinstance(dataset_knowledge, DatasetKnowledge):
            raise TypeError("dataset_knowledge must be DatasetKnowledge")

        validation = validate_analytical_plan(plan, dataset_knowledge)
        if validation.status != "READY":
            return _failure(
                plan, len(dataframe), "INVALID", "INVALID_PLAN",
                "; ".join(validation.errors) or validation.message,
            )
        missing = sorted({
            column
            for column in [
                *[item.column for item in plan.measures if item.column],
                *[item.column for item in plan.dimensions],
                *[item.column for item in plan.temporal],
                *[item.column for item in plan.filters],
            ]
            if column not in dataframe.columns
        })
        if missing:
            return _failure(
                plan, len(dataframe), "INVALID", "MISSING_COLUMN",
                f"Dataframe is missing required columns: {', '.join(missing)}.",
            )
        if plan.operation == "EXPLORATORY":
            return _failure(
                plan, len(dataframe), "UNSUPPORTED", "UNSUPPORTED_OPERATION",
                "Exploratory execution is pending a future analysis strategy.",
                result_type="EXPLORATORY_PLACEHOLDER",
            )

        try:
            working, filter_nulls, filter_invalid = self._apply_filters(plan, dataframe, dataset_knowledge)
            if working.empty:
                coverage = ExecutionCoverage(
                    source_rows=len(dataframe), filtered_rows=0, usable_rows=0,
                    excluded_null_rows=filter_nulls,
                    excluded_invalid_rows=filter_invalid,
                )
                return _failure(
                    plan, len(dataframe), "EMPTY", "EMPTY_RESULT",
                    "No rows remain after applying the typed filters.", coverage=coverage,
                )
            result = self._execute_operation(
                plan, working, len(dataframe), filter_nulls, filter_invalid,
                dataset_knowledge,
            )
        except Exception as exc:
            return _failure(
                plan, len(dataframe), "EXECUTION_ERROR", "EXECUTION_ERROR",
                f"Deterministic execution failed: {type(exc).__name__}.",
            )

        try:
            verification = verify_result(result, plan)
        except Exception as exc:
            return _failure(
                plan, len(dataframe), "EXECUTION_ERROR", "EXECUTION_ERROR",
                f"Result verification failed: {type(exc).__name__}.",
            )
        status = "INVALID" if verification.status == "FAILED" else result.status
        return result.model_copy(update={"status": status, "verification": verification})

    def _apply_filters(
        self,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
        knowledge: DatasetKnowledge,
    ) -> tuple[pd.DataFrame, int, int]:
        working = dataframe.copy(deep=False)
        nulls = 0
        invalid = 0
        for item in plan.filters:
            source = working[item.column]
            column = knowledge.get_column(item.column)
            assert column is not None
            if item.value_type in {"DATE", "DATETIME"}:
                comparable = pd.to_datetime(source, errors="coerce")
                values = item.value if isinstance(item.value, list) else [item.value]
                typed_values = [pd.Timestamp(value) for value in values]
            elif item.value_type == "NUMBER":
                comparable = pd.to_numeric(source, errors="coerce")
                values = item.value if isinstance(item.value, list) else [item.value]
                typed_values = [float(value) for value in values]
            else:
                comparable = source
                values = item.value if isinstance(item.value, list) else [item.value]
                typed_values = values
            valid = comparable.notna()
            nulls += int(source.isna().sum())
            invalid += int((source.notna() & ~valid).sum())
            mask = self._filter_mask(comparable, item, typed_values) & valid
            working = working.loc[mask]
        return working, nulls, invalid

    @staticmethod
    def _filter_mask(series: pd.Series, item: PlanFilter, values: list[Any]) -> pd.Series:
        first = values[0]
        if item.operator == "EQUALS":
            return series == first
        if item.operator == "NOT_EQUALS":
            return series != first
        if item.operator == "IN":
            return series.isin(values)
        if item.operator == "NOT_IN":
            return ~series.isin(values)
        if item.operator in {"GREATER_THAN", "AFTER"}:
            return series > first
        if item.operator == "GREATER_THAN_OR_EQUAL":
            return series >= first
        if item.operator in {"LESS_THAN", "BEFORE"}:
            return series < first
        if item.operator == "LESS_THAN_OR_EQUAL":
            return series <= first
        if item.operator == "BETWEEN":
            return series.between(values[0], values[1], inclusive="both")
        raise ValueError("Unsupported typed filter operator")

    def _execute_operation(
        self,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
        source_rows: int,
        filter_nulls: int,
        filter_invalid: int,
        knowledge: DatasetKnowledge,
    ) -> VerifiedResult:
        if plan.operation in {"AGGREGATE", "COUNT", "STATISTIC"} and not plan.dimensions:
            return self._scalar(plan, dataframe, source_rows, filter_nulls, filter_invalid)
        if plan.operation in {"GROUP_AGGREGATE", "COUNT", "RANK", "TOP_N", "BOTTOM_N", "PART_TO_WHOLE"}:
            return self._grouped(plan, dataframe, source_rows, filter_nulls, filter_invalid)
        if plan.operation == "TREND":
            return self._trend(plan, dataframe, source_rows, filter_nulls, filter_invalid)
        if plan.operation == "DISTRIBUTION":
            return self._distribution(plan, dataframe, source_rows, filter_nulls, filter_invalid, knowledge)
        if plan.operation == "RELATIONSHIP":
            return self._relationship(plan, dataframe, source_rows, filter_nulls, filter_invalid)
        return _failure(plan, source_rows, "UNSUPPORTED", "UNSUPPORTED_OPERATION", f"Operation {plan.operation} is unsupported.")

    def _scalar(self, plan, dataframe, source_rows, filter_nulls, filter_invalid) -> VerifiedResult:
        expression = plan.measures[0]
        nulls = filter_nulls
        invalid = filter_invalid
        details: dict[str, Any] = {}
        if expression.kind == "COUNT_ROWS":
            value, usable = len(dataframe), len(dataframe)
        elif expression.kind in {"COUNT_NON_NULL", "COUNT_DISTINCT"}:
            series = dataframe[expression.column]
            valid = series.notna()
            usable = int(valid.sum())
            nulls += len(series) - usable
            value = usable if expression.kind == "COUNT_NON_NULL" else int(series[valid].nunique(dropna=True))
        else:
            numeric, valid, own_nulls, own_invalid = _numeric(dataframe[expression.column])
            nulls += own_nulls
            invalid += own_invalid
            usable = int(valid.sum())
            if usable == 0:
                return self._empty(plan, source_rows, len(dataframe), nulls, invalid, "No usable numeric values remain.")
            values = numeric[valid]
            if expression.kind == "STATISTIC" and expression.operation == "STDDEV":
                if usable < 2:
                    return self._empty(plan, source_rows, len(dataframe), nulls, invalid, "Sample standard deviation requires at least two values.")
                value = float(values.std(ddof=1))
                details["definition"] = "sample standard deviation (ddof=1)"
            elif expression.kind == "STATISTIC" and expression.operation == "RANGE":
                value = float(values.max() - values.min())
                details["definition"] = "maximum minus minimum"
            else:
                value = _aggregate(values, expression.operation)
        result_type = "STATISTIC" if expression.kind == "STATISTIC" else "SCALAR"
        coverage = ExecutionCoverage(
            source_rows=source_rows, filtered_rows=len(dataframe), usable_rows=usable,
            excluded_null_rows=nulls, excluded_invalid_rows=invalid,
        )
        payload = {"scalar": value, "coverage": coverage.model_dump()}
        return VerifiedResult(
            result_id=_result_id(plan.plan_id, payload), plan_id=plan.plan_id,
            status="READY", result_type=result_type, scalar=_json_scalar(value),
            schema=ResultSchema(fields=[ResultField(name="value", dtype="number", role="VALUE")]),
            coverage=coverage, execution_metadata=_metadata(plan, details=details),
        )

    def _computed_groups(self, plan, dataframe) -> tuple[list[dict[str, Any]], int, int, int, dict[str, Any]]:
        dimension = plan.dimensions[0].column
        expression = plan.measures[0]
        dimension_missing = dataframe[dimension].isna()
        dimension_valid = ~dimension_missing
        nulls = int(dimension_missing.sum())
        invalid = 0
        if expression.kind == "COUNT_ROWS":
            valid = dimension_valid
            grouped = dataframe.loc[valid].groupby(dimension, dropna=True, sort=False).size()
            usable = int(valid.sum())
        elif expression.kind in {"COUNT_NON_NULL", "COUNT_DISTINCT"}:
            measure_missing = dataframe[expression.column].isna()
            measure_valid = ~measure_missing
            valid = dimension_valid & measure_valid
            nulls = int((dimension_missing | measure_missing).sum())
            grouped_obj = dataframe.loc[valid].groupby(dimension, dropna=True, sort=False)[expression.column]
            grouped = grouped_obj.count() if expression.kind == "COUNT_NON_NULL" else grouped_obj.nunique(dropna=True)
            usable = int(valid.sum())
        else:
            numeric, finite, own_nulls, own_invalid = _numeric(dataframe[expression.column])
            measure_missing = dataframe[expression.column].isna()
            nulls = int((dimension_missing | measure_missing).sum())
            invalid = int((dimension_valid & ~measure_missing & ~finite).sum())
            valid = dimension_valid & finite
            frame = pd.DataFrame({dimension: dataframe.loc[valid, dimension], "__value": numeric[valid]})
            grouped_obj = frame.groupby(dimension, dropna=True, sort=False)["__value"]
            function = {
                "SUM": grouped_obj.sum, "MEAN": grouped_obj.mean,
                "MEDIAN": grouped_obj.median, "MIN": grouped_obj.min,
                "MAX": grouped_obj.max,
            }[expression.operation]
            grouped = function()
            usable = int(valid.sum())
        rows = [
            {dimension: _json_scalar(group), "value": int(value) if expression.kind.startswith("COUNT") else float(value)}
            for group, value in grouped.items()
            if pd.notna(value) and math.isfinite(float(value))
        ]
        rows.sort(key=lambda row: _stable_key(row[dimension]))
        details: dict[str, Any] = {"group_count": len(rows)}
        if expression.kind == "AGGREGATION" and expression.operation == "SUM":
            details["grouped_total"] = float(sum(row["value"] for row in rows))
            details["aggregate_total"] = float(frame["__value"].sum())
        return rows, usable, nulls, invalid, details

    def _grouped(self, plan, dataframe, source_rows, filter_nulls, filter_invalid) -> VerifiedResult:
        rows, usable, nulls, invalid, details = self._computed_groups(plan, dataframe)
        nulls += filter_nulls
        invalid += filter_invalid
        if not rows:
            return self._empty(plan, source_rows, len(dataframe), nulls, invalid, "No usable groups remain.")
        dimension = plan.dimensions[0].column
        warnings: list[str] = []
        if plan.operation in {"RANK", "TOP_N", "BOTTOM_N"}:
            descending = plan.sort[0].direction == "DESC"
            rows.sort(key=lambda row: (-float(row["value"]) if descending else float(row["value"]), _stable_key(row[dimension])))
            if plan.operation == "RANK":
                winning_value = rows[0]["value"]
                rows = [row for row in rows if math.isclose(float(row["value"]), float(winning_value), rel_tol=1e-12, abs_tol=1e-12)]
                details["explicit_user_limit"] = False
                details["tied_winners"] = len(rows)
            else:
                details["explicit_user_limit"] = True
                details["requested_limit"] = plan.limit
                rows = rows[:plan.limit]
        if plan.operation == "PART_TO_WHOLE":
            if any(float(row["value"]) < 0 for row in rows):
                return _failure(plan, source_rows, "INVALID", "NEGATIVE_CONTRIBUTION", "Part-to-whole does not accept negative contributions.")
            denominator = float(sum(row["value"] for row in rows))
            if denominator <= 0:
                return _failure(plan, source_rows, "INVALID", "ZERO_DENOMINATOR", "Part-to-whole requires a positive total.")
            rows = [{**row, "share_of_total": float(row["value"]) / denominator} for row in rows]
            details["denominator"] = denominator
        fields = [
            ResultField(name=dimension, dtype="scalar", role="DIMENSION"),
            ResultField(name="value", dtype="number", role="MEASURE"),
        ]
        if plan.operation == "PART_TO_WHOLE":
            fields.append(ResultField(name="share_of_total", dtype="number", role="SHARE"))
        coverage = ExecutionCoverage(
            source_rows=source_rows, filtered_rows=len(dataframe), usable_rows=usable,
            excluded_null_rows=nulls, excluded_invalid_rows=invalid,
        )
        payload = {"rows": rows, "coverage": coverage.model_dump()}
        return VerifiedResult(
            result_id=_result_id(plan.plan_id, payload), plan_id=plan.plan_id,
            status="READY", result_type="TABLE", rows=rows,
            schema=ResultSchema(fields=fields), coverage=coverage,
            execution_metadata=_metadata(plan, details=details), warnings=warnings,
        )

    def _trend(self, plan, dataframe, source_rows, filter_nulls, filter_invalid) -> VerifiedResult:
        temporal = plan.temporal[0]
        parsed = pd.to_datetime(dataframe[temporal.column], errors="coerce")
        temporal_nulls = int(dataframe[temporal.column].isna().sum())
        temporal_invalid = int((dataframe[temporal.column].notna() & parsed.isna()).sum())
        expression = plan.measures[0]
        numeric, numeric_valid, numeric_nulls, numeric_invalid = _numeric(dataframe[expression.column])
        valid = parsed.notna() & numeric_valid
        if not valid.any():
            return self._empty(plan, source_rows, len(dataframe), filter_nulls + temporal_nulls + numeric_nulls, filter_invalid + temporal_invalid + numeric_invalid, "No valid temporal/measure pairs remain.")
        frequency = {"YEAR":"Y", "QUARTER":"Q", "MONTH":"M", "WEEK":"W-SUN", "DAY":"D", "HOUR":"h"}.get(temporal.time_granularity)
        bucket = (
            parsed[valid].dt.to_period(frequency).dt.start_time
            if frequency is not None else parsed[valid]
        )
        frame = pd.DataFrame({"__time": bucket, "__value": numeric[valid]})
        grouped_obj = frame.groupby("__time", sort=True)["__value"]
        grouped = {"SUM":grouped_obj.sum, "MEAN":grouped_obj.mean, "MEDIAN":grouped_obj.median, "MIN":grouped_obj.min, "MAX":grouped_obj.max}[expression.operation]()
        rows = [{temporal.column: value.isoformat(), "value": float(number)} for value, number in grouped.items()]
        coverage = ExecutionCoverage(
            source_rows=source_rows, filtered_rows=len(dataframe), usable_rows=int(valid.sum()),
            excluded_null_rows=filter_nulls + int((dataframe[temporal.column].isna() | dataframe[expression.column].isna()).sum()),
            excluded_invalid_rows=filter_invalid + int((dataframe[temporal.column].notna() & parsed.isna()).sum()) + int((parsed.notna() & dataframe[expression.column].notna() & ~numeric_valid).sum()),
        )
        details = {"point_count": len(rows), "first_value": rows[0]["value"], "last_value": rows[-1]["value"]}
        return VerifiedResult(
            result_id=_result_id(plan.plan_id, rows), plan_id=plan.plan_id,
            status="READY", result_type="SERIES", rows=rows,
            schema=ResultSchema(fields=[ResultField(name=temporal.column, dtype="datetime", role="TEMPORAL"), ResultField(name="value", dtype="number", role="MEASURE")]),
            coverage=coverage, execution_metadata=_metadata(plan, details=details),
        )

    def _distribution(self, plan, dataframe, source_rows, filter_nulls, filter_invalid, knowledge) -> VerifiedResult:
        expression = plan.measures[0]
        column = expression.column
        column_knowledge = knowledge.get_column(column)
        assert column_knowledge is not None
        if is_numeric_eligible(column_knowledge):
            numeric, valid, nulls, invalid = _numeric(dataframe[column])
            values = numeric[valid].astype(float)
            if values.empty:
                return self._empty(plan, source_rows, len(dataframe), filter_nulls + nulls, filter_invalid + invalid, "No numeric distribution values remain.")
            bin_count = min(int(math.ceil(math.log2(len(values)) + 1)), 50) if len(values) > 1 else 1
            counts, edges = np.histogram(values.to_numpy(), bins=bin_count)
            rows = [{"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]
            fields = [ResultField(name="bin_start", dtype="number", role="BIN"), ResultField(name="bin_end", dtype="number", role="BIN"), ResultField(name="count", dtype="integer", role="COUNT")]
            details = {"distribution_kind":"numeric_bins", "bin_strategy":"Sturges, capped at 50", "bin_count":bin_count}
            usable = int(valid.sum())
        elif is_category_eligible(column_knowledge):
            valid = dataframe[column].notna()
            counts = dataframe.loc[valid, column].value_counts(dropna=True, sort=False)
            rows = [{column:_json_scalar(value), "count":int(count)} for value, count in counts.items()]
            rows.sort(key=lambda row: (-int(row["count"]), _stable_key(row[column])))
            fields = [ResultField(name=column, dtype="scalar", role="DIMENSION"), ResultField(name="count", dtype="integer", role="COUNT")]
            details = {"distribution_kind":"categorical_frequency"}
            usable, nulls, invalid = int(valid.sum()), int((~valid).sum()), 0
        else:
            return _failure(plan, source_rows, "INVALID", "TYPE_MISMATCH", f"Column {column} is not distribution-compatible.", result_type="DISTRIBUTION")
        coverage = ExecutionCoverage(source_rows=source_rows, filtered_rows=len(dataframe), usable_rows=usable, excluded_null_rows=filter_nulls + nulls, excluded_invalid_rows=filter_invalid + invalid)
        return VerifiedResult(result_id=_result_id(plan.plan_id, rows), plan_id=plan.plan_id, status="READY", result_type="DISTRIBUTION", rows=rows, schema=ResultSchema(fields=fields), coverage=coverage, execution_metadata=_metadata(plan, details=details))

    def _relationship(self, plan, dataframe, source_rows, filter_nulls, filter_invalid) -> VerifiedResult:
        first, second = plan.measures[0].column, plan.measures[1].column
        first_values, first_valid, first_nulls, first_invalid = _numeric(dataframe[first])
        second_values, second_valid, second_nulls, second_invalid = _numeric(dataframe[second])
        valid = first_valid & second_valid
        rows = [{first:float(first_values.loc[index]), second:float(second_values.loc[index])} for index in dataframe.index[valid]]
        if not rows:
            return self._empty(plan, source_rows, len(dataframe), filter_nulls + first_nulls + second_nulls, filter_invalid + first_invalid + second_invalid, "No aligned valid pairs remain.", result_type="RELATIONSHIP")
        coverage = ExecutionCoverage(source_rows=source_rows, filtered_rows=len(dataframe), usable_rows=int(valid.sum()), excluded_null_rows=filter_nulls + int((dataframe[[first, second]].isna().any(axis=1)).sum()), excluded_invalid_rows=filter_invalid + int((dataframe[[first, second]].notna().all(axis=1) & ~valid).sum()))
        return VerifiedResult(result_id=_result_id(plan.plan_id, rows), plan_id=plan.plan_id, status="READY", result_type="RELATIONSHIP", rows=rows, schema=ResultSchema(fields=[ResultField(name=first, dtype="number", role="MEASURE"), ResultField(name=second, dtype="number", role="MEASURE")]), coverage=coverage, execution_metadata=_metadata(plan, details={"paired_rows":len(rows), "excluded_rows":len(dataframe)-len(rows)}))

    def _empty(self, plan, source_rows, filtered_rows, nulls, invalid, message, *, result_type="TABLE") -> VerifiedResult:
        coverage = ExecutionCoverage(source_rows=source_rows, filtered_rows=filtered_rows, usable_rows=0, excluded_null_rows=nulls, excluded_invalid_rows=invalid)
        return _failure(plan, source_rows, "EMPTY", "EMPTY_RESULT", message, result_type=result_type, coverage=coverage)


__all__ = ["PandasAnalyticalExecutor"]
