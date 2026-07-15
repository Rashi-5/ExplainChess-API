"""Mint an API key for a customer from the command line.

Usage:
    python -m scripts.mint_key --email coach@club.com --quota 500

Prints the raw key ONCE — store it immediately; only its hash is persisted.
Runs against whatever DATABASE_URL points to (Postgres in prod, SQLite in dev).
"""

from __future__ import annotations

import argparse

from app.db import SessionLocal, create_customer_with_key, init_db


def main() -> None:
    ap = argparse.ArgumentParser(description="Mint an ExplainChess API key.")
    ap.add_argument("--email", required=True)
    ap.add_argument("--tier", default="pilot")
    ap.add_argument("--quota", type=int, default=0, help="Daily call quota (0 = unlimited).")
    args = ap.parse_args()

    init_db()
    session = SessionLocal()
    try:
        customer, raw = create_customer_with_key(
            session, email=args.email, tier=args.tier, daily_quota=args.quota
        )
    finally:
        session.close()

    print(f"customer_id : {customer.id}")
    print(f"email       : {customer.email}")
    print(f"tier        : {customer.tier}  quota/day: {args.quota or 'unlimited'}")
    print(f"API KEY     : {raw}")
    print("\n^ Store this key now — it is not recoverable.")


if __name__ == "__main__":
    main()
