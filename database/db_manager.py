"""
Database Manager for handling database connections and sessions
"""
from contextlib import contextmanager
from typing import Generator, Optional
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize database manager

        Args:
            db_url: Database URL. If None, uses settings
        """
        self.db_url = db_url or settings.DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the database engine"""
        try:
            # For SQLite, use StaticPool to avoid threading issues
            if self.db_url.startswith("sqlite"):
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                    echo=settings.DEBUG
                )
            else:
                self.engine = create_engine(
                    self.db_url,
                    echo=settings.DEBUG
                )

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            logger.info("Database engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session

        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            logger.error(f"Session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        """Create all database tables"""
        try:
            from .models import Base
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def drop_tables(self) -> None:
        """Drop all database tables"""
        try:
            from .models import Base
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    def close(self) -> None:
        """Close the database engine"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")

    def health_check(self) -> bool:
        """Check database health"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
