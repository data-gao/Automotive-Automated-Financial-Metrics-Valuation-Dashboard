"""Modular data ingestion pipeline for automotive OEMs and Tier-1 suppliers."""

from __future__ import annotations

import argparse
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    PROCESSED_DATA_DIR,
    PeerConfig,
    load_peers_config,
)
from src.normalizer import convert_amount, fetch_fx_rates_to_eur

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    """Convert numeric values safely, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _extract_statement_item(
    df: pd.DataFrame,
    candidate_keys: List[str],
    col_idx: int = 0,
) -> Optional[float]:
    """Extract the first matching line item from a yfinance statement DataFrame."""
    if df is None or df.empty or col_idx >= len(df.columns):
        return None
    index_lower = {str(idx).strip().lower(): idx for idx in df.index}
    for key in candidate_keys:
        matched = index_lower.get(key.lower())
        if matched is not None:
            val = _safe_float(df.loc[matched].iloc[col_idx])
            if val is not None:
                return val
    return None


class BaseIngestionPipeline(ABC):
    """Abstract base class for financial data ingestion pipelines.

    Designed for extensibility: Phase 1 uses `YFinanceIngestionPipeline` for
    listed OEMs; Phase 2 can subclass this for Tier-1 suppliers or custom
    annual-report parsers (e.g., ZF Friedrichshafen private benchmarks).
    """

    @abstractmethod
    def fetch_peer_data(
        self, peer: PeerConfig, fx_to_eur: Dict[str, float]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Fetch latest summary metrics and multi-year historicals for a peer."""


class YFinanceIngestionPipeline(BaseIngestionPipeline):
    """Automated ingestion pipeline backed by Yahoo Finance (`yfinance`)."""

    def fetch_peer_data(
        self, peer: PeerConfig, fx_to_eur: Dict[str, float]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        logger.info("Fetching %s (%s)...", peer.name, peer.ticker)
        tk = yf.Ticker(peer.ticker)
        info = tk.info or {}

        fin = tk.financials
        bs = tk.balance_sheet
        cf = tk.cashflow

        # Determine currencies: trading vs. financial reporting
        trading_ccy = (info.get("currency") or peer.trading_currency).upper()
        reporting_ccy = (info.get("financialCurrency") or peer.reporting_currency).upper()
        if trading_ccy not in fx_to_eur:
            trading_ccy = peer.trading_currency
        if reporting_ccy not in fx_to_eur:
            reporting_ccy = peer.reporting_currency

        def to_eur_trade(val: Optional[float]) -> Optional[float]:
            return convert_amount(val, trading_ccy, "EUR", fx_to_eur)

        def to_eur_report(val: Optional[float]) -> Optional[float]:
            return convert_amount(val, reporting_ccy, "EUR", fx_to_eur)

        # Market & Trading Data (in Trading Currency -> EUR)
        price_local = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        market_cap_eur = to_eur_trade(_safe_float(info.get("marketCap")))
        beta = _safe_float(info.get("beta"))
        forward_pe = _safe_float(info.get("forwardPE"))
        trailing_pe = _safe_float(info.get("trailingPE"))
        price_to_book = _safe_float(info.get("priceToBook"))
        shares_outstanding = _safe_float(info.get("sharesOutstanding"))

        # 1-Year Price Return & Annualized Volatility
        return_1y = None
        volatility_1y = None
        try:
            hist = tk.history(period="1y")
            if not hist.empty and len(hist) > 10:
                closes = hist["Close"].dropna()
                return_1y = float((closes.iloc[-1] / closes.iloc[0]) - 1.0)
                daily_ret = closes.pct_change().dropna()
                volatility_1y = float(daily_ret.std() * np.sqrt(252))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Price history fetch failed for %s: %s", peer.ticker, exc)

        # Latest Fiscal Year Financial Statements (in Reporting Currency -> EUR)
        revenue = to_eur_report(
            _extract_statement_item(fin, ["Total Revenue", "Operating Revenue"])
        ) or to_eur_report(_safe_float(info.get("totalRevenue")))

        gross_profit = to_eur_report(_extract_statement_item(fin, ["Gross Profit"]))
        ebit = to_eur_report(
            _extract_statement_item(fin, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
        )
        ebitda = to_eur_report(
            _extract_statement_item(fin, ["EBITDA", "Normalized EBITDA"])
        )
        da = to_eur_report(
            _extract_statement_item(
                cf,
                ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
            )
        )
        if ebitda is None and ebit is not None and da is not None:
            ebitda = ebit + da
        if ebitda is None:
            ebitda = to_eur_report(_safe_float(info.get("ebitda")))

        net_income = to_eur_report(
            _extract_statement_item(fin, ["Net Income", "Net Income Common Stockholders"])
        )
        rd_expense = to_eur_report(
            _extract_statement_item(fin, ["Research And Development", "Research Development"])
        )
        interest_expense = to_eur_report(
            _extract_statement_item(fin, ["Interest Expense", "Interest Expense Non Operating"])
        )
        tax_provision = to_eur_report(_extract_statement_item(fin, ["Tax Provision"]))
        pretax_income = to_eur_report(_extract_statement_item(fin, ["Pretax Income"]))

        # Balance Sheet
        total_debt = to_eur_report(
            _extract_statement_item(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
        ) or to_eur_report(_safe_float(info.get("totalDebt")))

        cash_and_sti = to_eur_report(
            _extract_statement_item(
                bs,
                [
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                ],
            )
        ) or to_eur_report(_safe_float(info.get("totalCash")))

        total_equity = to_eur_report(
            _extract_statement_item(
                bs,
                ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
            )
        )
        minority_interest = to_eur_report(
            _extract_statement_item(bs, ["Minority Interest"])
        ) or 0.0

        # Cash Flow
        ocf = to_eur_report(
            _extract_statement_item(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        )
        capex_raw = to_eur_report(
            _extract_statement_item(cf, ["Capital Expenditure", "Purchase Of PPE"])
        )
        capex = abs(capex_raw) if capex_raw is not None else None
        fcf = to_eur_report(_extract_statement_item(cf, ["Free Cash Flow"]))
        if fcf is None and ocf is not None and capex is not None:
            fcf = ocf - capex
        if fcf is None:
            fcf = to_eur_report(_safe_float(info.get("freeCashflow")))

        # Enterprise Value (EUR)
        enterprise_value_eur = None
        if market_cap_eur is not None:
            debt_val = total_debt or 0.0
            cash_val = cash_and_sti or 0.0
            enterprise_value_eur = market_cap_eur + debt_val + minority_interest - cash_val

        summary = {
            "ticker": peer.ticker,
            "name": peer.name,
            "category": peer.category,
            "region": peer.region,
            "country": peer.country,
            "segment": peer.segment,
            "investment_angle": peer.investment_angle,
            "trading_currency": trading_ccy,
            "reporting_currency": reporting_ccy,
            "price_local": price_local,
            "price_eur": to_eur_trade(price_local),
            "shares_outstanding": shares_outstanding,
            "market_cap_eur": market_cap_eur,
            "enterprise_value_eur": enterprise_value_eur,
            "beta": beta,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "price_to_book": price_to_book,
            "return_1y": return_1y,
            "volatility_1y": volatility_1y,
            "revenue_eur": revenue,
            "gross_profit_eur": gross_profit,
            "ebitda_eur": ebitda,
            "ebit_eur": ebit,
            "net_income_eur": net_income,
            "rd_expense_eur": rd_expense,
            "da_eur": da,
            "interest_expense_eur": abs(interest_expense) if interest_expense is not None else None,
            "tax_provision_eur": tax_provision,
            "pretax_income_eur": pretax_income,
            "total_debt_eur": total_debt,
            "cash_eur": cash_and_sti,
            "total_equity_eur": total_equity,
            "minority_interest_eur": minority_interest,
            "ocf_eur": ocf,
            "capex_eur": capex,
            "fcf_eur": fcf,
        }

        # Multi-Year Historical Extraction
        historical_rows: List[Dict[str, Any]] = []
        if fin is not None and not fin.empty:
            for col_idx, col_date in enumerate(fin.columns):
                fiscal_year = str(col_date)[:4]
                rev_y = to_eur_report(
                    _extract_statement_item(fin, ["Total Revenue", "Operating Revenue"], col_idx)
                )
                ebit_y = to_eur_report(
                    _extract_statement_item(fin, ["EBIT", "Operating Income"], col_idx)
                )
                ebitda_y = to_eur_report(
                    _extract_statement_item(fin, ["EBITDA", "Normalized EBITDA"], col_idx)
                )
                ni_y = to_eur_report(
                    _extract_statement_item(fin, ["Net Income", "Net Income Common Stockholders"], col_idx)
                )
                rd_y = to_eur_report(
                    _extract_statement_item(fin, ["Research And Development", "Research Development"], col_idx)
                )
                fcf_y = to_eur_report(
                    _extract_statement_item(cf, ["Free Cash Flow"], col_idx)
                )
                historical_rows.append(
                    {
                        "ticker": peer.ticker,
                        "name": peer.name,
                        "region": peer.region,
                        "category": peer.category,
                        "fiscal_year": fiscal_year,
                        "revenue_eur": rev_y,
                        "ebitda_eur": ebitda_y,
                        "ebit_eur": ebit_y,
                        "net_income_eur": ni_y,
                        "rd_expense_eur": rd_y,
                        "fcf_eur": fcf_y,
                    }
                )

        return summary, historical_rows


def run_ingestion(
    categories: Optional[List[str]] = None,
    include_disabled: bool = False,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full ingestion pipeline and persist processed datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    peers = load_peers_config(categories=categories, include_disabled=include_disabled)
    fx_to_eur = fetch_fx_rates_to_eur()
    pipeline = YFinanceIngestionPipeline()

    summaries: List[Dict[str, Any]] = []
    historicals: List[Dict[str, Any]] = []

    for peer in peers:
        try:
            summary, hist = pipeline.fetch_peer_data(peer, fx_to_eur)
            summaries.append(summary)
            historicals.extend(hist)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to ingest %s (%s): %s", peer.name, peer.ticker, exc)

    summary_df = pd.DataFrame(summaries)
    hist_df = pd.DataFrame(historicals)

    summary_df.to_parquet(output_dir / "peers_summary.parquet", index=False)
    summary_df.to_csv(output_dir / "peers_summary.csv", index=False)

    if not hist_df.empty:
        hist_df = hist_df.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
        hist_df.to_parquet(output_dir / "historical_financials.parquet", index=False)
        hist_df.to_csv(output_dir / "historical_financials.csv", index=False)

    metadata = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "fx_to_eur": fx_to_eur,
        "peer_count": len(summary_df),
        "tickers": summary_df["ticker"].tolist() if not summary_df.empty else [],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Ingestion complete: %d peers saved to %s", len(summary_df), output_dir
    )
    return summary_df, hist_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automotive peer financial ingestion.")
    parser.add_argument(
        "--include-tier1",
        action="store_true",
        help="Include disabled categories such as Tier-1 suppliers.",
    )
    args = parser.parse_args()
    run_ingestion(include_disabled=args.include_tier1)
