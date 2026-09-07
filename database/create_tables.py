from database.base import Base
from database.connection import engine

from database.models.dataset import Dataset
from database.models.dataset_profile import DatasetProfile
from database.models.column_profile import ColumnProfile


def create_tables():
    Base.metadata.create_all(bind=engine)

    print("LYNX database tables created successfully.")


if __name__ == "__main__":
    create_tables()