import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class DatasetProfile(Base):
    """
    Stores deterministic profiling evidence
    for an entire dataset.
    """

    __tablename__ = "dataset_profiles"

    profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    row_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    column_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    total_missing_cells: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    overall_missing_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    exact_duplicate_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    duplicate_row_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    dtype_distribution: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )