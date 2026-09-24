"""Deterministic Visualization Request Parser — Pipeline 3, Stage 7.

Stage 7 of Pipeline 3: converts natural language user requests for visualizations
(e.g., "Show total sales by category", "Plot revenue over time", "Histogram of price")
into structured, deterministic VisualizationRequest objects.

Design Principles:
* Fully deterministic: no LLMs, no embeddings, no nondeterminism, no timestamps.
* Auditable: every matched column, extracted aggregation, and inferred chart type
  generates an explicit factual parser signal.
* Conservative column matching: exact normalized matching, token matching, and
  transparent semantic aliases with fixed deterministic tie-breaking.
* Explicit vs Inferred distinction: tracks whether the user explicitly named a
  chart type or whether it was inferred from the analytical request structure.
"""

from __future__ import annotations

import re
from typing import Sequence

import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_columns,
)
from pipelines.pipeline3_results.visualization.schemas import (
    DatasetClassification,
    RequestAggregation,
    VisualizationIntent,
    VisualizationRequest,
    VisualizationType,
)

# ---------------------------------------------------------------------------
# Deterministic Vocabularies
# ---------------------------------------------------------------------------

# Chart type keywords mapped to VisualizationType, sorted by phrase length desc
_CHART_TYPE_VOCABULARY: list[tuple[str, VisualizationType]] = [
    # HISTOGRAM
    ("frequency distribution", "HISTOGRAM"),
    ("distribution of", "HISTOGRAM"),
    ("histogram", "HISTOGRAM"),
    ("hist", "HISTOGRAM"),
    # COUNT_BAR
    ("category counts", "COUNT_BAR"),
    ("frequency chart", "COUNT_BAR"),
    ("frequency bar", "COUNT_BAR"),
    ("count chart", "COUNT_BAR"),
    ("count bar", "COUNT_BAR"),
    # SCATTER
    ("correlation plot", "SCATTER"),
    ("correlation chart", "SCATTER"),
    ("scatter plot", "SCATTER"),
    ("scatter graph", "SCATTER"),
    ("scatter chart", "SCATTER"),
    ("scatterplot", "SCATTER"),
    ("scatter", "SCATTER"),
    # LINE
    ("time series plot", "LINE"),
    ("time series chart", "LINE"),
    ("time series", "LINE"),
    ("trend line", "LINE"),
    ("line chart", "LINE"),
    ("line graph", "LINE"),
    ("line plot", "LINE"),
    ("lines", "LINE"),
    ("line", "LINE"),
    # BAR
    ("column chart", "BAR"),
    ("column graph", "BAR"),
    ("bar chart", "BAR"),
    ("bar graph", "BAR"),
    ("bar plot", "BAR"),
    ("bars", "BAR"),
    ("bar", "BAR"),
]

# Aggregation keywords mapped to RequestAggregation, sorted by phrase length desc
_AGGREGATION_VOCABULARY: list[tuple[str, RequestAggregation]] = [
    ("number of", "COUNT"),
    ("total count", "COUNT"),
    ("how many", "COUNT"),
    ("frequencies", "COUNT"),
    ("frequency", "COUNT"),
    ("num of", "COUNT"),
    ("counts", "COUNT"),
    ("count", "COUNT"),
    ("overall sum", "SUM"),
    ("cumulative", "SUM"),
    ("total", "SUM"),
    ("sum", "SUM"),
    ("average", "MEAN"),
    ("mean", "MEAN"),
    ("avg", "MEAN"),
    ("midpoint", "MEDIAN"),
    ("median", "MEDIAN"),
    ("minimum", "MIN"),
    ("smallest", "MIN"),
    ("lowest", "MIN"),
    ("min", "MIN"),
    ("maximum", "MAX"),
    ("greatest", "MAX"),
    ("highest", "MAX"),
    ("largest", "MAX"),
    ("max", "MAX"),
]

# Intent keywords mapped to VisualizationIntent
_INTENT_VOCABULARY: list[tuple[str, VisualizationIntent]] = [
    ("frequency distribution", "DISTRIBUTION"),
    ("distribution", "DISTRIBUTION"),
    ("spread", "DISTRIBUTION"),
    ("proportion", "DISTRIBUTION"),
    ("comparison", "COMPARISON"),
    ("compare", "COMPARISON"),
    ("difference", "COMPARISON"),
    ("breakdown", "COMPARISON"),
    ("over time", "TREND"),
    ("progression", "TREND"),
    ("timeline", "TREND"),
    ("temporal", "TREND"),
    ("monthly", "TREND"),
    ("yearly", "TREND"),
    ("daily", "TREND"),
    ("trend", "TREND"),
    ("relationship", "RELATIONSHIP"),
    ("correlation", "RELATIONSHIP"),
    ("association", "RELATIONSHIP"),
]

# Semantic aliases for common measure concepts
_MEASURE_ALIASES: dict[str, list[str]] = {
    "sales": ["total_price", "sales_amount", "total_amount", "sales", "revenue", "price"],
    "revenue": ["total_price", "sales_amount", "total_amount", "revenue", "sales", "price"],
    "price": ["total_price", "unit_price", "price", "sales_amount", "amount"],
    "cost": ["total_price", "unit_price", "cost", "price", "amount"],
    "profit": ["profit", "net_income", "margin", "earnings"],
    "quantity": ["quantity", "order_quantity", "count", "units", "items"],
    "rating": ["rating", "score", "performance_score", "satisfaction"],
    "time": ["order_date", "transaction_date", "timestamp", "date", "created_at"],
    "date": ["order_date", "transaction_date", "timestamp", "date", "created_at"],
}


# ---------------------------------------------------------------------------
# Normalization & Tokenization Helpers
# ---------------------------------------------------------------------------

def _normalize_string(text: str) -> str:
    """Normalize text by converting to lowercase and replacing hyphens/underscores with spaces."""
    cleaned = re.sub(r"[_\-]+", " ", text.casefold())
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return " ".join(cleaned.split())


def _extract_word_tokens(text: str) -> list[str]:
    """Extract individual alphanumeric word tokens in lowercase."""
    return re.findall(r"\b\w+\b", text.casefold())


# ---------------------------------------------------------------------------
# Deterministic Column Matcher
# ---------------------------------------------------------------------------

class ColumnMatch:
    """A matched column candidate with score and match details."""

    def __init__(
        self,
        column_name: str,
        match_type: str,
        tier: int,
        matched_text: str,
        start_pos: int,
    ) -> None:
        self.column_name = column_name
        self.match_type = match_type
        self.tier = tier  # 1 = exact normalized, 2 = exact token, 3 = alias/semantic
        self.matched_text = matched_text
        self.start_pos = start_pos

    def __repr__(self) -> str:
        return f"<ColumnMatch {self.column_name} (tier={self.tier}, type={self.match_type})>"


def _find_column_matches(
    request_text: str,
    columns: Sequence[str],
    classification: DatasetClassification | None = None,
) -> list[ColumnMatch]:
    """Deterministically find all column matches in request_text.

    Tie-breaking hierarchy (documented):
    1. Exact normalized column match (e.g. 'pizza category' -> 'pizza_category')
    2. Exact token match (e.g. 'quantity' -> 'quantity')
    3. Semantic alias match (e.g. 'sales' -> 'total_price')
    4. Shortest column name length
    5. Alphabetical column name
    """
    matches: list[ColumnMatch] = []
    norm_request = _normalize_string(request_text)
    request_tokens = set(_extract_word_tokens(request_text))

    matched_col_names: set[str] = set()

    # Pass 1: Multi-word exact normalized column match
    # Sort columns by normalized length descending so longer compound names match first
    sorted_cols = sorted(columns, key=lambda c: len(_normalize_string(c)), reverse=True)
    for col in sorted_cols:
        norm_col = _normalize_string(col)
        # Check whole-phrase match using regex word boundaries
        pattern = r"\b" + re.escape(norm_col) + r"\b"
        m = re.search(pattern, norm_request)
        if m:
            matches.append(
                ColumnMatch(
                    column_name=col,
                    match_type="exact_normalized",
                    tier=1,
                    matched_text=m.group(0),
                    start_pos=m.start(),
                )
            )
            matched_col_names.add(col)

    # Pass 2: Exact token match against request tokens
    for col in sorted_cols:
        if col in matched_col_names:
            continue
        col_norm = _normalize_string(col)
        col_tokens = col_norm.split()
        # If single-word column matches an exact token
        if len(col_tokens) == 1 and col_tokens[0] in request_tokens:
            pattern = r"\b" + re.escape(col_tokens[0]) + r"\b"
            m = re.search(pattern, norm_request)
            pos = m.start() if m else 999
            matches.append(
                ColumnMatch(
                    column_name=col,
                    match_type="exact_token",
                    tier=2,
                    matched_text=col_tokens[0],
                    start_pos=pos,
                )
            )
            matched_col_names.add(col)
        # If distinct unique token of a multi-word column appears in request
        elif len(col_tokens) > 1:
            for tok in col_tokens:
                if len(tok) >= 4 and tok in request_tokens:
                    # Check if token is unique to this column across all columns
                    matching_cols = [c for c in columns if tok in _normalize_string(c).split()]
                    if len(matching_cols) == 1:
                        pattern = r"\b" + re.escape(tok) + r"\b"
                        m = re.search(pattern, norm_request)
                        pos = m.start() if m else 999
                        matches.append(
                            ColumnMatch(
                                column_name=col,
                                match_type="unique_subtoken",
                                tier=2,
                                matched_text=tok,
                                start_pos=pos,
                            )
                        )
                        matched_col_names.add(col)
                        break

    # Pass 3: Semantic alias match (e.g. 'sales', 'revenue', 'time')
    for alias, candidate_names in _MEASURE_ALIASES.items():
        if alias in request_tokens:
            # Check if any candidate column exists in the dataset
            matched_alias_col: str | None = None
            for cname in candidate_names:
                for col in columns:
                    if col not in matched_col_names and (col.casefold() == cname.casefold() or cname.casefold() in col.casefold()):
                        matched_alias_col = col
                        break
                if matched_alias_col:
                    break

            if matched_alias_col:
                pattern = r"\b" + re.escape(alias) + r"\b"
                m = re.search(pattern, norm_request)
                pos = m.start() if m else 999
                matches.append(
                    ColumnMatch(
                        column_name=matched_alias_col,
                        match_type="semantic_alias",
                        tier=3,
                        matched_text=alias,
                        start_pos=pos,
                    )
                )
                matched_col_names.add(matched_alias_col)

    # Sort matches by position in request text to preserve natural phrase ordering
    matches.sort(key=lambda m: (m.start_pos, m.tier, len(m.column_name), m.column_name))
    return matches


# ---------------------------------------------------------------------------
# Main Public Function: parse_visualization_request
# ---------------------------------------------------------------------------

def parse_visualization_request(
    request_text: str,
    dataframe: pd.DataFrame,
    classification: DatasetClassification | None = None,
) -> VisualizationRequest:
    """Parse a natural-language visualization request into a VisualizationRequest.

    Parameters
    ----------
    request_text:
        Raw user text query (e.g., "Show total sales by category").
    dataframe:
        The target cleaned/prepared pandas DataFrame.
    classification:
        Optional Stage 1 DatasetClassification. Computed automatically if omitted.

    Returns
    -------
    VisualizationRequest
        Parsed deterministic specification with extracted chart type, columns,
        aggregation, intent, and factual audit signals.

    Raises
    ------
    TypeError
        If request_text is not a string or dataframe is not a pandas DataFrame.
    """
    if not isinstance(request_text, str):
        raise TypeError(f"request_text must be a string, got {type(request_text).__name__}")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")

    raw_request = request_text
    stripped_text = request_text.strip()
    norm_text = _normalize_string(stripped_text)
    signals: list[str] = []

    if not stripped_text:
        signals.append("empty request string")
        return VisualizationRequest(
            raw_request=raw_request,
            requested_chart_type=None,
            is_explicit_chart_type=False,
            x_column=None,
            y_column=None,
            aggregation=None,
            intent=None,
            requested_columns=[],
            parser_signals=signals,
        )

    if classification is None:
        classification = classify_columns(dataframe)

    # -----------------------------------------------------------------------
    # 1. Chart Type Recognition
    # -----------------------------------------------------------------------
    requested_chart_type: VisualizationType | None = None
    is_explicit_chart_type = False

    for phrase, ctype in _CHART_TYPE_VOCABULARY:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, norm_text):
            requested_chart_type = ctype
            is_explicit_chart_type = True
            signals.append(f"explicit chart type detected: '{phrase}' → {ctype}")
            break

    # -----------------------------------------------------------------------
    # 2. Column Matching (performed before aggregation to identify column spans)
    # -----------------------------------------------------------------------
    column_matches = _find_column_matches(
        request_text=stripped_text,
        columns=list(dataframe.columns),
        classification=classification,
    )

    requested_columns: list[str] = []
    for cm in column_matches:
        if cm.column_name not in requested_columns:
            requested_columns.append(cm.column_name)
            signals.append(
                f"matched request phrase '{cm.matched_text}' to column '{cm.column_name}' via {cm.match_type}"
            )

    # -----------------------------------------------------------------------
    # 3. Aggregation Recognition (prioritizes standalone aggregations over column tokens)
    # -----------------------------------------------------------------------
    detected_aggregation: RequestAggregation | None = None
    standalone_aggs: list[tuple[int, str, RequestAggregation]] = []
    subtoken_aggs: list[tuple[int, str, RequestAggregation]] = []

    for phrase, agg in _AGGREGATION_VOCABULARY:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        for m in re.finditer(pattern, norm_text):
            is_inside_col = any(
                cm.start_pos <= m.start() and m.end() <= (cm.start_pos + len(cm.matched_text))
                for cm in column_matches
            )
            if is_inside_col:
                subtoken_aggs.append((m.start(), phrase, agg))
            else:
                standalone_aggs.append((m.start(), phrase, agg))

    if standalone_aggs:
        # Order by appearance in request text, first modifier wins
        standalone_aggs.sort(key=lambda item: item[0])
        earliest_pos, earliest_phrase, earliest_agg = standalone_aggs[0]
        detected_aggregation = earliest_agg
        signals.append(f"aggregation detected: '{earliest_phrase}' → {earliest_agg}")
    elif subtoken_aggs:
        subtoken_aggs.sort(key=lambda item: item[0])
        earliest_pos, earliest_phrase, earliest_agg = subtoken_aggs[0]
        detected_aggregation = earliest_agg
        signals.append(f"aggregation derived from column token: '{earliest_phrase}' → {earliest_agg}")

    # -----------------------------------------------------------------------
    # 4. Intent Recognition
    # -----------------------------------------------------------------------
    detected_intent: VisualizationIntent | None = None
    for phrase, intent_val in _INTENT_VOCABULARY:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, norm_text):
            detected_intent = intent_val
            signals.append(f"analytical intent detected: '{phrase}' → {intent_val}")
            break

    # -----------------------------------------------------------------------
    # 5. Dimension & Measure Assignment (X and Y Columns)
    # -----------------------------------------------------------------------
    x_column: str | None = None
    y_column: str | None = None

    # Check for relational prepositions: 'by', 'over', 'versus', 'vs', 'between'
    by_match = re.search(r"\bby\s+([a-zA-Z0-9_\-\s]+)", norm_text)
    over_match = re.search(r"\bover\s+([a-zA-Z0-9_\-\s]+)", norm_text)
    vs_match = re.search(r"\b(?:versus|vs|against)\s+([a-zA-Z0-9_\-\s]+)", norm_text)
    between_match = re.search(r"\bbetween\s+([a-zA-Z0-9_\-\s]+)\s+and\s+([a-zA-Z0-9_\-\s]+)", norm_text)

    # Prepositional Resolution
    if by_match and len(requested_columns) >= 2:
        # "Y by X" pattern (e.g. "sales by category", "total_price by pizza_size")
        # In English data phrasing, the column after 'by' is the grouping dimension (X)
        phrase_after_by = by_match.group(1).strip()
        by_col = None
        for col in requested_columns:
            if _normalize_string(col) in phrase_after_by:
                by_col = col
                break
        if by_col:
            x_column = by_col
            other_cols = [c for c in requested_columns if c != by_col]
            y_column = other_cols[0] if other_cols else None
            signals.append(f"resolved 'Y by X' structure: x='{x_column}', y='{y_column}'")

    elif over_match and len(requested_columns) >= 2:
        # "Y over X" pattern (e.g. "sales over time", "total_price over order_date")
        phrase_after_over = over_match.group(1).strip()
        over_col = None
        for col in requested_columns:
            if _normalize_string(col) in phrase_after_over:
                over_col = col
                break
        if over_col:
            x_column = over_col
            other_cols = [c for c in requested_columns if c != over_col]
            y_column = other_cols[0] if other_cols else None
            signals.append(f"resolved 'Y over X' structure: x='{x_column}', y='{y_column}'")

    elif vs_match and len(requested_columns) >= 2:
        # "X versus Y" pattern
        phrase_after_vs = vs_match.group(1).strip()
        vs_col = None
        for col in requested_columns:
            if _normalize_string(col) in phrase_after_vs:
                vs_col = col
                break
        if vs_col:
            y_column = vs_col
            other_cols = [c for c in requested_columns if c != vs_col]
            x_column = other_cols[0] if other_cols else None
            signals.append(f"resolved 'X versus Y' structure: x='{x_column}', y='{y_column}'")

    elif between_match and len(requested_columns) >= 2:
        p1 = between_match.group(1).strip()
        p2 = between_match.group(2).strip()
        for col in requested_columns:
            ncol = _normalize_string(col)
            if ncol in p1 and x_column is None:
                x_column = col
            elif ncol in p2 and y_column is None:
                y_column = col
        if not x_column or not y_column:
            x_column, y_column = requested_columns[0], requested_columns[1]

    # Fallback Assignment by Column Roles
    if x_column is None and len(requested_columns) == 1:
        x_column = requested_columns[0]
        y_column = None
    elif x_column is None and len(requested_columns) >= 2:
        c1 = requested_columns[0]
        c2 = requested_columns[1]
        r1 = classification.get_column(c1)
        r2 = classification.get_column(c2)
        role1 = r1.role if r1 else "UNKNOWN"
        role2 = r2.role if r2 else "UNKNOWN"

        # If one is dimension (CATEGORY, BOOLEAN, TEMPORAL) and other is MEASURE:
        if role1 in ("CATEGORY", "BOOLEAN", "TEMPORAL") and role2 == "MEASURE":
            x_column, y_column = c1, c2
        elif role2 in ("CATEGORY", "BOOLEAN", "TEMPORAL") and role1 == "MEASURE":
            x_column, y_column = c2, c1
        else:
            # Default order
            x_column, y_column = c1, c2

    # -----------------------------------------------------------------------
    # 6. Chart Type & Intent Inference (when not explicit)
    # -----------------------------------------------------------------------
    if requested_chart_type is None:
        if x_column and y_column:
            rx = classification.get_column(x_column)
            ry = classification.get_column(y_column)
            role_x = rx.role if rx else "UNKNOWN"
            role_y = ry.role if ry else "UNKNOWN"

            if role_x in ("CATEGORY", "BOOLEAN") and role_y == "MEASURE":
                requested_chart_type = "BAR"
                detected_intent = detected_intent or "COMPARISON"
                signals.append("chart type inferred from category + measure comparison request")
            elif role_x == "TEMPORAL" and role_y == "MEASURE":
                requested_chart_type = "LINE"
                detected_intent = detected_intent or "TREND"
                signals.append("chart type inferred from temporal + measure trend request")
            elif role_x == "MEASURE" and role_y == "MEASURE":
                requested_chart_type = "SCATTER"
                detected_intent = detected_intent or "RELATIONSHIP"
                signals.append("chart type inferred from measure + measure relationship request")

        elif x_column and not y_column:
            rx = classification.get_column(x_column)
            role_x = rx.role if rx else "UNKNOWN"

            if role_x == "MEASURE":
                requested_chart_type = "HISTOGRAM"
                detected_intent = detected_intent or "DISTRIBUTION"
                signals.append("chart type inferred from single measure distribution request")
            elif role_x in ("CATEGORY", "BOOLEAN"):
                requested_chart_type = "COUNT_BAR"
                detected_intent = detected_intent or "DISTRIBUTION"
                signals.append("chart type inferred from single category frequency request")

    # Set default intent based on chart type if still unset
    if detected_intent is None and requested_chart_type is not None:
        if requested_chart_type in ("HISTOGRAM", "COUNT_BAR"):
            detected_intent = "DISTRIBUTION"
        elif requested_chart_type == "BAR":
            detected_intent = "COMPARISON"
        elif requested_chart_type == "LINE":
            detected_intent = "TREND"
        elif requested_chart_type == "SCATTER":
            detected_intent = "RELATIONSHIP"

    return VisualizationRequest(
        raw_request=raw_request,
        requested_chart_type=requested_chart_type,
        is_explicit_chart_type=is_explicit_chart_type,
        x_column=x_column,
        y_column=y_column,
        aggregation=detected_aggregation,
        intent=detected_intent,
        requested_columns=requested_columns,
        parser_signals=signals,
    )
