"""Per-key usage metering and daily-quota enforcement.

`check_quota` is called before doing work (rejects with 429 if the customer's
daily allowance is already spent). `record_usage` is called after a successful
call to increment the day's counters — the single source of truth for both
quota checks and (later) Stripe billing.
"""

from __future__ import annotations

import datetime as dt

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Customer, Usage


def _today() -> dt.date:
    return dt.datetime.utcnow().date()


def _usage_row(session: Session, key_hash: str, day: dt.date) -> Usage:
    row = session.scalar(
        select(Usage).where(Usage.key_hash == key_hash, Usage.day == day)
    )
    if row is None:
        row = Usage(key_hash=key_hash, day=day, analyze_calls=0, batch_puzzles=0)
        session.add(row)
        session.flush()
    return row


def _spent_today(session: Session, key_hash: str, day: dt.date) -> int:
    row = session.scalar(
        select(Usage).where(Usage.key_hash == key_hash, Usage.day == day)
    )
    if row is None:
        return 0
    # Both analyze calls and per-puzzle batch work count toward the quota.
    return row.analyze_calls + row.batch_puzzles


def check_quota(session: Session, key_hash: str, customer: Customer, cost: int = 1) -> None:
    """Reject the request if this `cost` would exceed the customer's daily quota.
    A quota of 0 means unlimited."""
    if customer.daily_quota <= 0:
        return
    spent = _spent_today(session, key_hash, _today())
    if spent + cost > customer.daily_quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily quota exceeded ({customer.daily_quota} calls/day). "
                f"Already used {spent} today."
            ),
        )


def record_usage(
    session: Session, key_hash: str, *, analyze_calls: int = 0, batch_puzzles: int = 0
) -> None:
    row = _usage_row(session, key_hash, _today())
    row.analyze_calls += analyze_calls
    row.batch_puzzles += batch_puzzles
    session.commit()
