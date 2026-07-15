"""API-key authentication.

Each protected route depends on `require_key`, which validates the Bearer token
against the `api_keys` table and returns the resolved (ApiKey, Customer). The
admin key-minting route uses `require_admin` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ADMIN_TOKEN
from app.db import ApiKey, Customer, SessionLocal, hash_key


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@dataclass
class Principal:
    """The authenticated caller for a request."""
    key_hash: str
    customer: Customer


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Use 'Authorization: Bearer <api_key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header. Expected 'Bearer <api_key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1].strip()


def require_key(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> Principal:
    raw = _extract_bearer(authorization)
    key_hash = hash_key(raw)
    api_key = session.scalar(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.active.is_(True))
    )
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(key_hash=key_hash, customer=api_key.customer)


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Guards the key-minting endpoint. Constant-time-ish compare is unnecessary
    here (single fixed secret over TLS), but reject when unset to avoid an
    accidentally open admin surface."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin operations are disabled (ADMIN_TOKEN not set).",
        )
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token.",
        )
