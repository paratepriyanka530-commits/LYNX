import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class ColumnProfile(Base):
    """
    Stores deterministic profiling evidence
    for one dataset column.
    """

    __tablename__ = "column_profiles"

    column_profile_id: Mapped[uuid.UUID] = mapped_column(
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

    column_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    column_position: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    physical_dtype: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    row_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    non_null_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    missing_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    missing_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    unique_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    unique_ratio: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    repeated_unique_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    repeated_occurrences: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    repetition_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    sample_values: Mapped[list] = mapped_column(
        JSONB,
        nullable=False
    )

    numeric_statistics: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )