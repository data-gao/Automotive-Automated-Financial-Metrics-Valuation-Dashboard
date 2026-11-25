"""Cross-currency normalization and accounting alignment utilities."""

from __future__ import annotations

import logging
from typing import Dict

import yfinance as yf

from src.config import FALLBACK_FX_TO_EUR, FX_YAHOO_PAIRS, SUPPORTED_CURRENCIES

logger = logging.getLogger(__name__)


def fetch_fx_rates_to_eur() -> Dict[str, float]:
    """Fetch live spot FX rates converting 1 unit of each currency into EUR.

    Returns:
        Dictionary mapping currency code (e.g. 'USD', 'CNY', 'HKD', 'EUR')
        to its equivalent value in EUR.
    """
    rates: Dict[str, float] = {"EUR": 1.0}
    for ccy, pair in FX_YAHOO_PAIRS.items():
        try:
            ticker = yf.Ticker(pair)
            hist = ticker.history(period="5d")
            if not hist.empty:
                eur_in_ccy = float(hist["Close"].dropna().iloc[-1])
                if eur_in_ccy > 0:
                    rates[ccy] = 1.0 / eur_in_ccy
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("FX fetch failed for %s (%s): %s", ccy, pair, exc)

        rates[ccy] = FALLBACK_FX_TO_EUR[ccy]
        logger.info("Using fallback FX rate for %s -> EUR: %.4f", ccy, rates[ccy])

    return rates


def convert_amount(
    value: float | None,
    from_currency: str,
    to_currency: str,
    fx_to_eur: Dict[str, float],
) -> float | None:
    """Convert a monetary amount from `from_currency` to `to_currency` via EUR pivot."""
    if value is None:
        return None
    src = (from_currency or "EUR").upper()
    dst = (to_currency or "EUR").upper()
    if src == dst:
        return float(value)
    if src not in fx_to_eur or dst not in fx_to_eur:
        raise ValueError(f"Unsupported currency conversion: {src} -> {dst}")

    amount_in_eur = float(value) * fx_to_eur[src]
    return amount_in_eur / fx_to_eur[dst]


def get_fx_matrix(fx_to_eur: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Build a conversion multiplier matrix across all supported currencies."""
    matrix: Dict[str, Dict[str, float]] = {}
    for src in SUPPORTED_CURRENCIES:
        matrix[src] = {
            dst: convert_amount(1.0, src, dst, fx_to_eur) or 1.0
            for dst in SUPPORTED_CURRENCIES
        }
    return matrix
