"""Financial ratio, capital efficiency, and relative valuation calculation engine."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import FALLBACK_FX_TO_EUR
from src.normalizer import convert_amount

MONETARY_COLS_SUMMARY = [
    "price_eur",
    "market_cap_eur",
    "enterprise_value_eur",
    "revenue_eur",
    "gross_profit_eur",
    "ebitda_eur",
    "ebit_eur",
    "net_income_eur",
    "rd_expense_eur",
    "da_eur",
    "interest_expense_eur",
    "tax_provision_eur",
    "pretax_income_eur",
    "total_debt_eur",
    "cash_eur",
    "total_equity_eur",
    "minority_interest_eur",
    "ocf_eur",
    "capex_eur",
    "fcf_eur",
]

MONETARY_COLS_HIST = [
    "revenue_eur",
    "ebitda_eur",
    "ebit_eur",
    "net_income_eur",
    "rd_expense_eur",
    "fcf_eur",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise safe division returning NaN where denominator is zero or missing."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def compute_peer_metrics(
    df: pd.DataFrame,
    target_currency: str = "EUR",
    fx_to_eur: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute standardized financial ratios and relative valuation multiples.

    All monetary columns stored in EUR are converted to `target_currency`
    (with `_ccy` suffix or updated display columns in billions).
    """
    if df.empty:
        return df.copy()

    fx = fx_to_eur or FALLBACK_FX_TO_EUR
    mult = convert_amount(1.0, "EUR", target_currency, fx) or 1.0

    out = df.copy()
    for col in MONETARY_COLS_SUMMARY:
        if col in out.columns:
            new_col = col.replace("_eur", "_ccy")
            out[new_col] = pd.to_numeric(out[col], errors="coerce") * mult

    # Convenience columns in Billions of target currency
    for base in [
        "market_cap",
        "enterprise_value",
        "revenue",
        "ebitda",
        "ebit",
        "net_income",
        "rd_expense",
        "total_debt",
        "cash",
        "fcf",
    ]:
        ccy_col = f"{base}_ccy"
        if ccy_col in out.columns:
            out[f"{base}_bn"] = out[ccy_col] / 1e9

    # Net Debt
    debt = out["total_debt_ccy"].fillna(0.0)
    cash = out["cash_ccy"].fillna(0.0)
    out["net_debt_ccy"] = debt - cash
    out["net_debt_bn"] = out["net_debt_ccy"] / 1e9

    # Profitability Margins
    rev = out["revenue_ccy"]
    out["gross_margin"] = _safe_div(out["gross_profit_ccy"], rev)
    out["ebitda_margin"] = _safe_div(out["ebitda_ccy"], rev)
    out["ebit_margin"] = _safe_div(out["ebit_ccy"], rev)
    out["net_margin"] = _safe_div(out["net_income_ccy"], rev)
    out["fcf_margin"] = _safe_div(out["fcf_ccy"], rev)

    # Innovation & Reinvestment
    out["rd_intensity"] = _safe_div(out["rd_expense_ccy"], rev)
    out["capex_intensity"] = _safe_div(out["capex_ccy"], rev)

    # Effective Tax Rate (bounded between 15% and 35%, fallback 25%)
    raw_tax = _safe_div(out["tax_provision_ccy"], out["pretax_income_ccy"])
    out["effective_tax_rate"] = raw_tax.clip(lower=0.15, upper=0.35).fillna(0.25)

    # Capital Efficiency: ROIC & ROE
    nopat = out["ebit_ccy"] * (1.0 - out["effective_tax_rate"])
    invested_capital = debt + out["total_equity_ccy"] - cash
    out["roic"] = _safe_div(nopat, invested_capital.where(invested_capital > 0))
    out["roe"] = _safe_div(out["net_income_ccy"], out["total_equity_ccy"].where(out["total_equity_ccy"] > 0))

    # Leverage & Solvency
    ebitda_pos = out["ebitda_ccy"].where(out["ebitda_ccy"] > 0)
    out["net_debt_to_ebitda"] = _safe_div(out["net_debt_ccy"], ebitda_pos)
    out["debt_to_equity"] = _safe_div(debt, out["total_equity_ccy"].where(out["total_equity_ccy"] > 0))
    out["interest_coverage"] = _safe_div(
        out["ebit_ccy"], out["interest_expense_ccy"].where(out["interest_expense_ccy"] > 0)
    )

    # Relative Valuation Multiples
    ev = out["enterprise_value_ccy"]
    mcap = out["market_cap_ccy"]
    out["ev_to_ebitda"] = _safe_div(ev, ebitda_pos)
    out["ev_to_sales"] = _safe_div(ev, rev.where(rev > 0))
    out["fcf_yield"] = _safe_div(out["fcf_ccy"], mcap.where(mcap > 0))

    return out


def compute_historical_metrics(
    hist_df: pd.DataFrame,
    target_currency: str = "EUR",
    fx_to_eur: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Convert historical financials to target currency and compute YoY growth & margins."""
    if hist_df.empty:
        return hist_df.copy()

    fx = fx_to_eur or FALLBACK_FX_TO_EUR
    mult = convert_amount(1.0, "EUR", target_currency, fx) or 1.0

    out = hist_df.copy()
    for col in MONETARY_COLS_HIST:
        if col in out.columns:
            ccy_col = col.replace("_eur", "_ccy")
            out[ccy_col] = pd.to_numeric(out[col], errors="coerce") * mult
            out[col.replace("_eur", "_bn")] = out[ccy_col] / 1e9

    out = out.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    out["revenue_yoy"] = out.groupby("ticker")["revenue_ccy"].pct_change()
    out["ebitda_margin"] = _safe_div(out["ebitda_ccy"], out["revenue_ccy"])
    out["ebit_margin"] = _safe_div(out["ebit_ccy"], out["revenue_ccy"])
    out["net_margin"] = _safe_div(out["net_income_ccy"], out["revenue_ccy"])
    out["rd_intensity"] = _safe_div(out["rd_expense_ccy"], out["revenue_ccy"])

    return out
