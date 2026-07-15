"""Stripe metered-usage reporting — wired but OFF by default (V1).

When BILLING_ENABLED is false (the V1 default), `report_usage` is a no-op and
pilots are invoiced manually from the `usage` table. When enabled, each metered
unit is reported to the customer's Stripe metered subscription item.

This intentionally reports per-call rather than via a nightly batch job: it's
simpler for a low-volume pilot and needs no scheduler. Swap to an aggregated
daily push (sum the `usage` table) if per-call Stripe latency ever matters.
"""

from __future__ import annotations

import logging

from app.config import BILLING_ENABLED, STRIPE_API_KEY
from app.db import Customer

log = logging.getLogger("explainchess.billing")

_stripe = None
if BILLING_ENABLED:
    if not STRIPE_API_KEY:
        raise RuntimeError("BILLING_ENABLED is true but STRIPE_API_KEY is not set.")
    import stripe as _stripe_mod

    _stripe_mod.api_key = STRIPE_API_KEY
    _stripe = _stripe_mod


def report_usage(customer: Customer, units: int) -> None:
    """Report `units` of metered usage for `customer`. No-op unless billing is on.

    Failures are logged, never raised: a billing outage must not turn a
    successful analysis into a 500 for the customer. Under-reporting is
    reconciled from the `usage` table, which remains the source of truth.
    """
    if not BILLING_ENABLED or _stripe is None:
        return
    if units <= 0 or not customer.stripe_customer_id:
        return
    try:
        # Requires a metered subscription item mapped to this customer. Resolving
        # the subscription-item id is deployment-specific; store it on the
        # customer (add a column) or look it up here once pricing is set.
        _stripe.billing.MeterEvent.create(
            event_name="explainchess_analysis",
            payload={"value": str(units), "stripe_customer_id": customer.stripe_customer_id},
        )
    except Exception:  # noqa: BLE001 - billing must never break the request path
        log.exception("Stripe usage report failed for customer %s", customer.id)
