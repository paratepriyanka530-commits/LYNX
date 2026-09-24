"""Deterministic Visualization Request Validator — Pipeline 3, Stage 7.

Stage 7 of Pipeline 3: validates a VisualizationRequest against dataset column
intelligence and Stage 2 feasibility rules.

This module answers:
    "Is the requested visualization valid and structurally meaningful for this dataset?"

Possible outcomes:
* VALID: The request is complete, structurally compatible, and can be rendered.
* INVALID: The request contains semantic contradictions, forbidden roles
  (IDENTIFIER, DESCRIPTIVE_TEXT, UNKNOWN), incompatible chart/column combinations,
  missing temporals, or insufficient measures.
* NEEDS_CLARIFICATION: The request is ambiguous, lacks analytical columns, or
  matches multiple candidate measures.
"""

from __future__ import annotations

import pandas as pd

from pipelines.pipeline3_results.visualization.feasibility import (
    _MAX_CATEGORY_CARDINALITY,
    _MIN_USABLE_NUMERIC_ROWS,
    _is_category_usable,
    _is_measure_usable,
    _is_temporal_usable,
)
from pipelines.pipeline3_results.visualization.schemas import (
    DatasetClassification,
    FeasibilityResult,
    ValidatedVisualizationRequest,
    VisualizationRequest,
    VisualizationRequestStatus,
    VisualizationType,
)


def validate_visualization_request(
    dataframe: pd.DataFrame,
    classification: DatasetClassification,
    request: VisualizationRequest,
    feasibility: FeasibilityResult | None = None,
) -> ValidatedVisualizationRequest:
    """Validate a parsed VisualizationRequest against dataset characteristics.

    Parameters
    ----------
    dataframe:
        The target cleaned/prepared pandas DataFrame.
    classification:
        Stage 1 DatasetClassification for the dataset.
    request:
        Parsed VisualizationRequest from parse_visualization_request.
    feasibility:
        Optional Stage 2 FeasibilityResult.

    Returns
    -------
    ValidatedVisualizationRequest
        Validated specification with status VALID, INVALID, or NEEDS_CLARIFICATION,
        accompanying factual signals, errors (if invalid), and clarification options.

    Raises
    ------
    TypeError
        If inputs are of incorrect types.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")
    if not isinstance(classification, DatasetClassification):
        raise TypeError(f"classification must be a DatasetClassification, got {type(classification).__name__}")
    if not isinstance(request, VisualizationRequest):
        raise TypeError(f"request must be a VisualizationRequest, got {type(request).__name__}")

    signals: list[str] = list(request.parser_signals)
    errors: list[str] = []
    clarification_options: list[str] = []

    x_col_name = request.x_column
    y_col_name = request.y_column
    chart_type = request.requested_chart_type
    aggregation = request.aggregation
    intent = request.intent
    requested_cols = list(request.requested_columns)

    # -----------------------------------------------------------------------
    # 1. Ambiguity & Missing Information Checks (NEEDS_CLARIFICATION)
    # -----------------------------------------------------------------------
    # Case A: Completely empty or generic request ("make me a graph", "show chart", "compare these")
    if not x_col_name and not requested_cols and not chart_type:
        signals.append("no chart type specified")
        signals.append("no analytical columns specified")
        options = [c.name for c in classification.columns if c.role in ("MEASURE", "CATEGORY", "TEMPORAL")]
        return ValidatedVisualizationRequest(
            status="NEEDS_CLARIFICATION",
            chart_type=None,
            x_column=None,
            y_column=None,
            aggregation=aggregation,
            intent=intent,
            requested_columns=requested_cols,
            signals=signals,
            errors=[],
            clarification_options=options,
        )

    # Case B: Chart type requested without columns ("make a scatter plot", "bar chart", etc.)
    if chart_type is not None and not x_col_name and not requested_cols:
        if chart_type == "SCATTER":
            if len(classification.measures) < 2:
                errors.append("At least two MEASURE columns are required for a scatter plot.")
                signals.append(f"scatter plot rejected: dataset contains only {len(classification.measures)} measure(s)")
                return ValidatedVisualizationRequest(
                    status="INVALID",
                    chart_type=chart_type,
                    x_column=None,
                    y_column=None,
                    aggregation=aggregation,
                    intent="RELATIONSHIP",
                    requested_columns=[],
                    signals=signals,
                    errors=errors,
                    clarification_options=[],
                )
            signals.append("SCATTER plot requested without specifying columns")
            return ValidatedVisualizationRequest(
                status="NEEDS_CLARIFICATION",
                chart_type=chart_type,
                x_column=None,
                y_column=None,
                aggregation=aggregation,
                intent="RELATIONSHIP",
                requested_columns=[],
                signals=signals,
                errors=[],
                clarification_options=classification.measures,
            )
        elif chart_type == "LINE":
            if not classification.temporals:
                errors.append("No suitable temporal column was found for a time-series visualization.")
                signals.append("LINE chart rejected: dataset has no temporal columns")
                return ValidatedVisualizationRequest(
                    status="INVALID",
                    chart_type=chart_type,
                    x_column=None,
                    y_column=None,
                    aggregation=aggregation,
                    intent="TREND",
                    requested_columns=[],
                    signals=signals,
                    errors=errors,
                    clarification_options=[],
                )
            signals.append("LINE chart requested without specifying measure column")
            return ValidatedVisualizationRequest(
                status="NEEDS_CLARIFICATION",
                chart_type=chart_type,
                x_column=classification.temporals[0],
                y_column=None,
                aggregation=aggregation,
                intent="TREND",
                requested_columns=[],
                signals=signals,
                errors=[],
                clarification_options=classification.measures,
            )
        else:
            signals.append(f"{chart_type} requested without specifying analytical columns")
            options = [c.name for c in classification.columns if c.role in ("MEASURE", "CATEGORY", "TEMPORAL")]
            return ValidatedVisualizationRequest(
                status="NEEDS_CLARIFICATION",
                chart_type=chart_type,
                x_column=None,
                y_column=None,
                aggregation=aggregation,
                intent=intent,
                requested_columns=[],
                signals=signals,
                errors=[],
                clarification_options=options,
            )

    # -----------------------------------------------------------------------
    # 2. Dataset-Level Structural Prerequisite Checks
    # -----------------------------------------------------------------------
    if chart_type == "LINE" and not classification.temporals:
        errors.append("No suitable temporal column was found for a time-series visualization.")
        signals.append("LINE chart rejected: dataset has no temporal columns")
        return ValidatedVisualizationRequest(
            status="INVALID",
            chart_type=chart_type,
            x_column=x_col_name,
            y_column=y_col_name,
            aggregation=aggregation,
            intent="TREND",
            requested_columns=requested_cols,
            signals=signals,
            errors=errors,
            clarification_options=[],
        )

    if chart_type == "SCATTER" and len(classification.measures) < 2:
        errors.append("At least two MEASURE columns are required for a scatter plot.")
        signals.append(f"scatter plot rejected: dataset contains only {len(classification.measures)} measure(s)")
        return ValidatedVisualizationRequest(
            status="INVALID",
            chart_type=chart_type,
            x_column=x_col_name,
            y_column=y_col_name,
            aggregation=aggregation,
            intent="RELATIONSHIP",
            requested_columns=requested_cols,
            signals=signals,
            errors=errors,
            clarification_options=[],
        )

    # -----------------------------------------------------------------------
    # 3. Column Existence Verification
    # -----------------------------------------------------------------------
    if x_col_name is not None and x_col_name not in dataframe.columns:
        errors.append(f"Column '{x_col_name}' does not exist in dataset.")

    if y_col_name is not None and y_col_name not in dataframe.columns:
        errors.append(f"Column '{y_col_name}' does not exist in dataset.")

    if errors:
        return ValidatedVisualizationRequest(
            status="INVALID",
            chart_type=chart_type,
            x_column=x_col_name,
            y_column=y_col_name,
            aggregation=aggregation,
            intent=intent,
            requested_columns=requested_cols,
            signals=signals,
            errors=errors,
            clarification_options=[],
        )

    # -----------------------------------------------------------------------
    # 3. Forbidden Roles Verification (IDENTIFIER, DESCRIPTIVE_TEXT, UNKNOWN)
    # -----------------------------------------------------------------------
    x_col_class = classification.get_column(x_col_name) if x_col_name else None
    y_col_class = classification.get_column(y_col_name) if y_col_name else None

    x_role = x_col_class.role if x_col_class else None
    y_role = y_col_class.role if y_col_class else None

    # Check for two identifiers (e.g. "plot order_id against customer_id")
    if x_role == "IDENTIFIER" and y_role == "IDENTIFIER":
        errors.append("Both requested columns are IDENTIFIER fields and do not represent meaningful analytical measures.")
        signals.append(f"rejected: '{x_col_name}' and '{y_col_name}' are identifiers")
        return ValidatedVisualizationRequest(
            status="INVALID",
            chart_type=chart_type,
            x_column=x_col_name,
            y_column=y_col_name,
            aggregation=aggregation,
            intent=intent,
            requested_columns=requested_cols,
            signals=signals,
            errors=errors,
            clarification_options=[],
        )

    if x_col_name:
        if x_role == "IDENTIFIER":
            errors.append(f"'{x_col_name}' is classified as IDENTIFIER and cannot be used as an analytical dimension.")
        elif x_role == "DESCRIPTIVE_TEXT":
            errors.append(f"'{x_col_name}' is classified as DESCRIPTIVE_TEXT and cannot be used as a visualization dimension.")
        elif x_role == "UNKNOWN":
            errors.append(f"'{x_col_name}' is classified as UNKNOWN (constant or uninformative) and cannot be used in a visualization.")

    if y_col_name:
        if y_role == "IDENTIFIER":
            errors.append(f"'{y_col_name}' is classified as IDENTIFIER and cannot be used as an analytical measure.")
        elif y_role == "DESCRIPTIVE_TEXT":
            errors.append(f"'{y_col_name}' is classified as DESCRIPTIVE_TEXT and cannot be used as an analytical measure.")
        elif y_role == "UNKNOWN":
            errors.append(f"'{y_col_name}' is classified as UNKNOWN (constant or uninformative) and cannot be used in a visualization.")

    if errors:
        return ValidatedVisualizationRequest(
            status="INVALID",
            chart_type=chart_type,
            x_column=x_col_name,
            y_column=y_col_name,
            aggregation=aggregation,
            intent=intent,
            requested_columns=requested_cols,
            signals=signals,
            errors=errors,
            clarification_options=[],
        )

    # -----------------------------------------------------------------------
    # 4. Chart Structural Compatibility Checks
    # -----------------------------------------------------------------------
    # If no chart type resolved yet
    if chart_type is None:
        if x_col_name and not y_col_name:
            if x_role == "MEASURE":
                chart_type = "HISTOGRAM"
                intent = intent or "DISTRIBUTION"
            elif x_role in ("CATEGORY", "BOOLEAN"):
                chart_type = "COUNT_BAR"
                intent = intent or "DISTRIBUTION"
            else:
                errors.append(f"Single column '{x_col_name}' with role {x_role} does not support 1D visualization.")
        elif x_col_name and y_col_name:
            if x_role in ("CATEGORY", "BOOLEAN") and y_role == "MEASURE":
                chart_type = "BAR"
                intent = intent or "COMPARISON"
            elif x_role == "TEMPORAL" and y_role == "MEASURE":
                chart_type = "LINE"
                intent = intent or "TREND"
            elif x_role == "MEASURE" and y_role == "MEASURE":
                chart_type = "SCATTER"
                intent = intent or "RELATIONSHIP"
            else:
                errors.append(
                    f"Combination of x='{x_col_name}' ({x_role}) and y='{y_col_name}' ({y_role}) "
                    f"is not structurally supported for visualization."
                )

    if errors:
        return ValidatedVisualizationRequest(
            status="INVALID",
            chart_type=chart_type,
            x_column=x_col_name,
            y_column=y_col_name,
            aggregation=aggregation,
            intent=intent,
            requested_columns=requested_cols,
            signals=signals,
            errors=errors,
            clarification_options=[],
        )

    # Validate against specific chart type rules
    if chart_type == "HISTOGRAM":
        if not x_col_name:
            errors.append("HISTOGRAM requires a MEASURE column.")
        elif x_role != "MEASURE":
            errors.append(
                f"'{x_col_name}' is classified as {x_role} and cannot be used as a histogram measure. "
                f"HISTOGRAM requires a MEASURE column."
            )
        else:
            ok, reason = _is_measure_usable(dataframe, x_col_class)
            if not ok:
                errors.append(f"HISTOGRAM measure '{x_col_name}' is unusable: {reason}.")
            else:
                signals.append(f"HISTOGRAM validated on measure '{x_col_name}'")

    elif chart_type == "COUNT_BAR":
        if not x_col_name:
            errors.append("COUNT_BAR requires a CATEGORY or BOOLEAN column.")
        elif x_role not in ("CATEGORY", "BOOLEAN"):
            errors.append(
                f"'{x_col_name}' is classified as {x_role}. "
                f"COUNT_BAR requires a CATEGORY or BOOLEAN column."
            )
        else:
            limit = 2 if x_role == "BOOLEAN" else _MAX_CATEGORY_CARDINALITY
            ok, reason = _is_category_usable(dataframe, x_col_class, high_cardinality_limit=limit)
            if not ok:
                errors.append(f"COUNT_BAR dimension '{x_col_name}' is unusable: {reason}.")
            else:
                signals.append(f"COUNT_BAR validated on {x_role} '{x_col_name}'")

    elif chart_type == "BAR":
        if not x_col_name or not y_col_name:
            errors.append("BAR chart requires both a dimension (CATEGORY/BOOLEAN) and a measure (MEASURE).")
        else:
            if x_role not in ("CATEGORY", "BOOLEAN"):
                errors.append(f"BAR chart dimension '{x_col_name}' must be CATEGORY or BOOLEAN, got {x_role}.")
            else:
                limit = 2 if x_role == "BOOLEAN" else _MAX_CATEGORY_CARDINALITY
                ok, reason = _is_category_usable(dataframe, x_col_class, high_cardinality_limit=limit)
                if not ok:
                    errors.append(f"BAR chart dimension '{x_col_name}' is unusable: {reason}.")

            if y_role != "MEASURE":
                errors.append(f"BAR chart measure '{y_col_name}' must be MEASURE, got {y_role}.")
            else:
                ok, reason = _is_measure_usable(dataframe, y_col_class)
                if not ok:
                    errors.append(f"BAR chart measure '{y_col_name}' is unusable: {reason}.")

            if not errors:
                signals.append(f"BAR chart validated: dimension='{x_col_name}', measure='{y_col_name}'")

    elif chart_type == "LINE":
        if not classification.temporals:
            errors.append("No suitable temporal column was found for a time-series visualization.")
        elif not x_col_name:
            errors.append("LINE chart requires a TEMPORAL column for the x-axis.")
        elif x_role != "TEMPORAL":
            errors.append(f"LINE chart x-axis '{x_col_name}' must be TEMPORAL, got {x_role}.")
        elif not y_col_name or y_role != "MEASURE":
            errors.append(f"LINE chart measure '{y_col_name}' must be MEASURE, got {y_role}.")
        else:
            ok_x, reason_x = _is_temporal_usable(dataframe, x_col_class)
            ok_y, reason_y = _is_measure_usable(dataframe, y_col_class)
            if not ok_x:
                errors.append(f"LINE chart temporal column '{x_col_name}' is unusable: {reason_x}.")
            if not ok_y:
                errors.append(f"LINE chart measure column '{y_col_name}' is unusable: {reason_y}.")
            if not errors:
                signals.append(f"LINE chart validated: temporal='{x_col_name}', measure='{y_col_name}'")

    elif chart_type == "SCATTER":
        if len(classification.measures) < 2:
            errors.append("At least two MEASURE columns are required for a scatter plot.")
        elif not x_col_name or not y_col_name:
            errors.append("SCATTER plot requires two MEASURE columns.")
        elif x_col_name == y_col_name:
            errors.append("SCATTER plot requires two distinct MEASURE columns, got identical columns.")
        elif x_role != "MEASURE" or y_role != "MEASURE":
            errors.append(f"Both columns in a SCATTER plot must be MEASURE, got x={x_role}, y={y_role}.")
        else:
            ok_x, reason_x = _is_measure_usable(dataframe, x_col_class)
            ok_y, reason_y = _is_measure_usable(dataframe, y_col_class)
            if not ok_x:
                errors.append(f"SCATTER x-measure '{x_col_name}' is unusable: {reason_x}.")
            if not ok_y:
                errors.append(f"SCATTER y-measure '{y_col_name}' is unusable: {reason_y}.")
            if not errors:
                signals.append(f"SCATTER plot validated: x='{x_col_name}', y='{y_col_name}'")

    # -----------------------------------------------------------------------
    # 5. Final Result Construction
    # -----------------------------------------------------------------------
    if errors:
        final_status: VisualizationRequestStatus = "INVALID"
    else:
        final_status = "VALID"
        signals.append("visualization request successfully validated")

    return ValidatedVisualizationRequest(
        status=final_status,
        chart_type=chart_type,
        x_column=x_col_name,
        y_column=y_col_name,
        aggregation=aggregation,
        intent=intent,
        requested_columns=requested_cols,
        signals=signals,
        errors=errors,
        clarification_options=clarification_options,
    )
