import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# Validate Configuration
# ============================================================

required_variables = {
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
}


missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]


if missing_variables:
    raise RuntimeError(
        "Missing database environment variables: "
        + ", ".join(missing_variables)
    )


# ============================================================
# Database URL
# ============================================================

DATABASE_URL = (
    f"postgresql+psycopg://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# SQLAlchemy Engine
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# Session Factory
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# ============================================================
# Connection Test
# ============================================================

def test_database_connection() -> bool:
    """
    Test whether LYNX can connect to PostgreSQL.
    """

    try:

        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT 1")
            )

            value = result.scalar()

            if value == 1:
                print(
                    "LYNX database connection successful."
                )

                return True

    except Exception as exc:

        print(
            "LYNX database connection failed."
        )

        print(
            f"Reason: {exc}"
        )

        return False


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    test_database_connection()