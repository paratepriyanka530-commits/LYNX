from uuid import UUID
from typing import Any, Dict

from sqlalchemy.orm import Session

from database.models.dataset_profile import DatasetProfile
from database.models.column_profile import ColumnProfile


class ProfileRepository:
    """
    Handles persistence of dataset and column
    profiling evidence.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_profile(
        self,
        dataset_id: UUID,
        profile: Dict[str, Any]
    ) -> DatasetProfile:
        """
        Save a complete dataset profile and
        all associated column profiles.
        """

        dataset_profile = DatasetProfile(
            dataset_id=dataset_id,

            row_count=profile["shape"]["rows"],

            column_count=profile["shape"]["columns"],

            total_missing_cells=(
                profile["missingness"][
                    "total_missing_cells"
                ]
            ),

            overall_missing_percentage=(
                profile["missingness"][
                    "overall_missing_percentage"
                ]
            ),

            exact_duplicate_rows=(
                profile["duplicates"][
                    "exact_duplicate_rows"
                ]
            ),

            duplicate_row_percentage=(
                profile["duplicates"][
                    "percentage"
                ]
            ),

            dtype_distribution=(
                profile["dtype_distribution"]
            )
        )

        self.session.add(
            dataset_profile
        )

        # ----------------------------------------------------
        # Column profiles
        # ----------------------------------------------------

        for position, column in enumerate(
            profile["column_profiles"]
        ):

            column_profile = ColumnProfile(
                dataset_id=dataset_id,

                column_name=column[
                    "column_name"
                ],

                column_position=position,

                physical_dtype=column[
                    "physical_dtype"
                ],

                row_count=column[
                    "row_count"
                ],

                non_null_count=column[
                    "non_null_count"
                ],

                missing_count=column[
                    "missing"
                ]["count"],

                missing_percentage=column[
                    "missing"
                ]["percentage"],

                unique_count=column[
                    "uniqueness"
                ]["unique_count"],

                unique_ratio=column[
                    "uniqueness"
                ]["unique_ratio"],

                repeated_unique_count=column[
                    "repetition"
                ]["repeated_unique_count"],

                repeated_occurrences=column[
                    "repetition"
                ]["repeated_occurrences"],

                repetition_rate=column[
                    "repetition"
                ]["repetition_rate"],

                sample_values=column[
                    "sample_values"
                ],

                numeric_statistics=column.get(
                    "numeric_statistics"
                )
            )

            self.session.add(
                column_profile
            )

        # ----------------------------------------------------
        # Commit everything together
        # ----------------------------------------------------

        try:
            self.session.commit()

            self.session.refresh(
                dataset_profile
            )

        except Exception:
            self.session.rollback()
            raise

        return dataset_profile