import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class Dataset(Base):
    """
    Represents a dataset registered in LYNX.

    This table stores dataset metadata.
    The actual dataset file remains in storage.
    """

    __tablename__ = "datasets"

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # --------------------------------------------------------
    # File Information
    # --------------------------------------------------------

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    stored_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False
    )

    file_format: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )

    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False
    )

    # --------------------------------------------------------
    # Dataset Structure
    # --------------------------------------------------------

    row_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    column_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    # --------------------------------------------------------
    # Pipeline State
    # --------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="registered"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )