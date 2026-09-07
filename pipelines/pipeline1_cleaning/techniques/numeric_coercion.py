"""Whole-column numeric conversion with explicit preservation checks."""

import math
import re
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict


QUANTITATIVE_TYPES = {"numeric_measurement", "currency", "percentage"}
TARGET_DTYPES = {"integer": "Int64", "float": "Float64"}
MIN_NUMERIC_COVERAGE = 0.95


class CoerceDtypeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_dtype: Literal["integer", "float"]


def assess_numeric(series):
    parsed, non_null, invalid, leading_zero = [], 0, 0, 0
    for value in series.tolist():
        if pd.api.types.is_scalar(value) and pd.isna(value):
            parsed.append(None)
            continue
        non_null += 1
        number = None
        if isinstance(value, str):
            text = value.strip()
            if re.match(r"^[+-]?0[0-9]", text):
                leading_zero += 1
            if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
                try:
                    number = Decimal(text)
                except InvalidOperation:
                    pass
        elif isinstance(value, (Integral, Real, Decimal)) and not isinstance(value, bool):
            number = Decimal(str(value))
        if number is None or not number.is_finite():
            invalid += 1
            parsed.append(None)
        else:
            parsed.append(number)
    numbers = [number for number in parsed if number is not None]
    return parsed, {
        "non_null_count": non_null,
        "parseable_count": len(numbers), "invalid_count": invalid,
        "numeric_parse_coverage": len(numbers) / non_null if non_null else 0.0,
        "minimum_coverage": MIN_NUMERIC_COVERAGE,
        "required_safe_conversion_coverage": 1.0,
        "leading_zero_count": leading_zero,
        "suggested_target_dtype": (
            "integer" if numbers and all(n == n.to_integral_value() for n in numbers) else "float"
        ),
    }


def convert_numeric(series, parameters=None):
    parsed, stats = assess_numeric(series)
    if stats["leading_zero_count"]:
        raise ValueError("Leading-zero/code evidence blocks automatic numeric coercion.")
    if stats["numeric_parse_coverage"] < MIN_NUMERIC_COVERAGE:
        raise ValueError("Numeric coverage below 95%; column requires review.")
    if stats["invalid_count"]:
        raise ValueError("Every non-null value must convert safely (100%); column preserved.")
    supplied = dict(parameters or {})
    if "target_dtype" not in supplied:
        supplied["target_dtype"] = stats["suggested_target_dtype"]
    target = CoerceDtypeParameters.model_validate(supplied).target_dtype
    values = []
    for number in parsed:
        if number is None:
            values.append(pd.NA)
        elif target == "integer":
            if number != number.to_integral_value():
                raise ValueError("Integer conversion would lose decimal information.")
            if not Decimal(-(2**63)) <= number <= Decimal(2**63 - 1):
                raise ValueError("Integer value does not fit signed Int64.")
            values.append(int(number))
        else:
            value = float(number)
            # Reject overflow, underflow, and loss visible on decimal round-trip.
            if not math.isfinite(value) or Decimal(str(value)) != number:
                raise ValueError("Float64 conversion would lose numeric information.")
            values.append(value)
    stats["chosen_target_dtype"] = target
    return pd.Series(values, index=series.index, name=series.name, dtype=TARGET_DTYPES[target]), stats


def coerce_dtype(dataframe, target_columns, parameters):
    if len(target_columns) != 1:
        raise ValueError("coerce_dtype requires exactly one target column.")
    validated = CoerceDtypeParameters.model_validate(parameters)
    column = target_columns[0]
    converted, _ = convert_numeric(dataframe[column], validated.model_dump())
    dataframe[column] = converted
    return dataframe
