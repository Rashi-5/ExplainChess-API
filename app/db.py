"""Datastore: customers, API keys, and per-key daily usage.

SQLAlchemy 2.0 models over Postgres in production (`DATABASE_URL`), falling back
to a local SQLite file for dev/tests. API keys are stored **hashed** — the raw
key is shown once at mint time and never persisted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets

from sqlalchemy import (
    Boolean, Date, DateTime, ForeignKey, Integer, String, UniqueConstraint,
    create_engine, func, select,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker,
)

from app.config import DATABASE_URL

# SQLite needs check_same_thread=False since the pipeline runs requests across
# threads; Postgres ignores connect_args.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tier: Mapped[str] = mapped_column(String(32), default="pilot")
    # Daily quota (analyze-equivalent calls); batch counts per-puzzle. 0 = unlimited.
    daily_quota: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    keys: Mapped[list["ApiKey"]] = relationship(back_populates="customer")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    customer: Mapped["Customer"] = relationship(back_populates="keys")


class Usage(Base):
    __tablename__ = "usage"
    __table_args__ = (UniqueConstraint("key_hash", "day", name="uq_usage_key_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), index=True)
    day: Mapped[dt.date] = mapped_column(Date, index=True)
    analyze_calls: Mapped[int] = mapped_column(Integer, default=0)
    batch_puzzles: Mapped[int] = mapped_column(Integer, default=0)


# --- key helpers ------------------------------------------------------------

KEY_PREFIX = "eck_"  # ExplainChess key


def hash_key(raw: str) -> str:
    """SHA-256 of the raw key. Keys are high-entropy, so a plain hash (no salt)
    is fine — this is a lookup token, not a user password."""
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def init_db() -> None:
    """Create tables if absent. Safe to call at startup; for schema changes use
    a real migration tool (Alembic) later."""
    Base.metadata.create_all(engine)


def create_customer_with_key(
    session: Session, email: str, tier: str = "pilot", daily_quota: int = 0
) -> tuple[Customer, str]:
    """Create (or reuse) a customer and mint a fresh key. Returns the customer
    and the RAW key (shown once — only its hash is stored)."""
    customer = session.scalar(select(Customer).where(Customer.email == email))
    if customer is None:
        customer = Customer(email=email, tier=tier, daily_quota=daily_quota)
        session.add(customer)
        session.flush()

    raw = generate_key()
    session.add(ApiKey(key_hash=hash_key(raw), customer_id=customer.id))
    session.commit()
    return customer, raw
