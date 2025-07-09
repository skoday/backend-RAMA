from sqlalchemy import create_engine, TIMESTAMP
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from app.env_settings import settings


DATABASE_USERNAME = settings.database_username
DATABASE_PASSWORD = settings.database_password
DATABASE_HOSTNAME = settings.database_hostname
DATABASE_PORT = settings.database_port
DATABASE_NAME = settings.database_name


SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOSTNAME}:{DATABASE_PORT}/{DATABASE_NAME}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    type_annotation_map = {
        datetime: TIMESTAMP(timezone=True)
    }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()