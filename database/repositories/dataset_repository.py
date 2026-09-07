from pathlib import Path
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models.dataset import Dataset


class DatasetRepository:
    """
    Handles database operations related to datasets.
    """

    def __init__(self, session: Session):
        self.session = session

    def register_dataset(
        self,
        original_filename: str,
        stored_path: str,
        file_format: str,
        file_size_bytes: int,
        row_count: int,
        column_count: int,
        status: str = "registered"
    ) -> Dataset:
        """
        Register a new dataset in PostgreSQL.
        """

        dataset = Dataset(
            original_filename=original_filename,
            stored_path=stored_path,
            file_format=file_format,
            file_size_bytes=file_size_bytes,
            row_count=row_count,
            column_count=column_count,
            status=status
        )

        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)

        return dataset

    def get_by_id(
        self,
        dataset_id: UUID
    ) -> Optional[Dataset]:
        """
        Retrieve a dataset using its UUID.
        """

        statement = select(Dataset).where(
            Dataset.dataset_id == dataset_id
        )

        return self.session.scalar(statement)

    def get_all(self) -> list[Dataset]:
        """
        Retrieve all registered datasets.
        """

        statement = select(Dataset)

        return list(
            self.session.scalars(statement).all()
        )
    def update_status(
        self,
        dataset_id: UUID,
        status: str
    ) -> Dataset:
        """
        Update the processing status of a dataset.
        """

        dataset = self.get_by_id(
            dataset_id
        )

        if dataset is None:
            raise ValueError(
                f"Dataset not found: {dataset_id}"
            )

        dataset.status = status

        self.session.commit()
        self.session.refresh(dataset)

        return dataset