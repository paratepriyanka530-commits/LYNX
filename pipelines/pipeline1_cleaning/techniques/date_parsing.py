"""Conservative, explicit-format date parsing for visualization exports."""

import re
from datetime import datetime

import pandas as pd


OUTPUT_FORMATS = {
    "date": "YYYY-MM-DD",
    "datetime": "YYYY-MM-DD HH:MM:SS",
    "time": "HH:MM:SS",
}
MIN_PARSE_COVERAGE = 0.95
_OUTPUT_PATTERNS = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
    "HH:MM:SS": "%H:%M:%S",
}


def _interpret(value, kind):
    if not isinstance(value, str):
        return [], False
    text = value.strip()
    time_pattern = r"[0-9]{2}:[0-9]{2}(?::[0-9]{2})?"
    if kind == "time":
        if not re.fullmatch(time_pattern, text):
            return [], False
        formats = ["%H:%M", "%H:%M:%S"]
    else:
        suffix = ""
        date_text = text
        if kind == "datetime":
            match = re.fullmatch(r"(.+)([ T])(" + time_pattern + r")", text)
            if not match:
                return [], False
            date_text, separator, clock = match.groups()
            suffix = separator + ("%H:%M:%S" if clock.count(":") == 2 else "%H:%M")
        if re.fullmatch(r"[0-9]{4}([-\/])[0-9]{2}\1[0-9]{2}", date_text):
            separator = date_text[4]
            formats = [f"%Y{separator}%m{separator}%d" + suffix]
        elif re.fullmatch(r"[0-9]{2}([-\/])[0-9]{2}\1[0-9]{4}", date_text):
            separator = date_text[2]
            formats = [
                f"%d{separator}%m{separator}%Y" + suffix,
                f"%m{separator}%d{separator}%Y" + suffix,
            ]
        else:
            return [], False
    interpretations = set()
    for pattern in formats:
        try:
            interpretations.add(datetime.strptime(text, pattern))
        except ValueError:
            pass
    return list(interpretations), len(interpretations) > 1


def assess_dates(series, output_format):
    """Return a preservation-safe preview and coverage; never guess a locale."""
    if output_format not in _OUTPUT_PATTERNS:
        raise ValueError("Unsupported parse_dates output_format.")
    kind = next(key for key, value in OUTPUT_FORMATS.items() if value == output_format)
    values = []
    parseable = ambiguous = non_null = 0
    for value in series.tolist():
        if pd.api.types.is_scalar(value) and pd.isna(value):
            values.append(value)
            continue
        non_null += 1
        candidates, is_ambiguous = _interpret(value, kind)
        ambiguous += int(is_ambiguous)
        if len(candidates) == 1:
            parseable += 1
            values.append(candidates[0].strftime(_OUTPUT_PATTERNS[output_format]))
        else:
            values.append(value)
    coverage = parseable / non_null if non_null else 0.0
    stats = {
        "non_null_count": non_null, "parseable_count": parseable,
        "unparseable_count": non_null - parseable - ambiguous,
        "ambiguous_count": ambiguous, "parse_coverage": coverage,
        "minimum_coverage": MIN_PARSE_COVERAGE,
        "output_format": output_format,
    }
    return pd.Series(values, index=series.index, dtype=series.dtype, name=series.name), stats


def require_safe_dates(stats):
    if stats["ambiguous_count"]:
        raise ValueError("Ambiguous day/month values require review; column preserved.")
    if stats["parse_coverage"] < MIN_PARSE_COVERAGE:
        raise ValueError("parse_dates coverage is below 95%; column preserved.")


def parse_dates(dataframe, target_columns, parameters):
    if len(target_columns) != 1 or set(parameters) != {"output_format"}:
        raise ValueError("parse_dates requires one target and only output_format.")
    column = target_columns[0]
    preview, stats = assess_dates(dataframe[column], parameters["output_format"])
    require_safe_dates(stats)
    dataframe[column] = preview
    return dataframe
