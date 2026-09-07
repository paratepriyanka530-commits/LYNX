from typing import Any, Dict, List


def build_column_evidence(
    dataset_profile: Dict[str, Any],
    column_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a controlled evidence package for semantic analysis.

    This function does not infer meaning.
    It only selects relevant observed evidence.
    """

    return {
        "dataset_context": {
            "row_count": dataset_profile["shape"]["rows"],
            "column_count": dataset_profile["shape"]["columns"],
            "column_names": dataset_profile["column_names"],
        },

        "column": {
            "column_name": column_profile["column_name"],
            "physical_dtype": column_profile["physical_dtype"],

            "missing": {
                "count": column_profile["missing"]["count"],
                "percentage": column_profile["missing"]["percentage"],
            },

            "uniqueness": {
                "unique_count": column_profile["uniqueness"]["unique_count"],
                "unique_ratio": column_profile["uniqueness"]["unique_ratio"],
            },

            "repetition": {
                "repeated_unique_count": (
                    column_profile["repetition"][
                        "repeated_unique_count"
                    ]
                ),
                "repeated_occurrences": (
                    column_profile["repetition"][
                        "repeated_occurrences"
                    ]
                ),
                "repetition_rate": (
                    column_profile["repetition"][
                        "repetition_rate"
                    ]
                ),
            },

            "sample_values": column_profile["sample_values"],

            "numeric_statistics": column_profile.get(
                "numeric_statistics"
            ),
        },
    }


def build_dataset_evidence(
    dataset_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build dataset-level evidence for semantic analysis.
    """

    compact_columns: List[Dict[str, Any]] = []

    for column in dataset_profile["column_profiles"]:

        compact_columns.append(
            {
                "column_name": column["column_name"],
                "physical_dtype": column["physical_dtype"],
                "missing_percentage": (
                    column["missing"]["percentage"]
                ),
                "unique_ratio": (
                    column["uniqueness"]["unique_ratio"]
                ),
                "sample_values": (
                    column["sample_values"]
                ),
            }
        )

    return {
        "shape": dataset_profile["shape"],
        "column_names": dataset_profile["column_names"],
        "dtype_distribution": dataset_profile["dtype_distribution"],
        "missingness": dataset_profile["missingness"],
        "duplicates": dataset_profile["duplicates"],
        "columns": compact_columns,
    }