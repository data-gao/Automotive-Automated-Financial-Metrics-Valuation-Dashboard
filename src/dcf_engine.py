"""Discounted Cash Flow (DCF) valuation and 2D sensitivity analysis engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import DCFDefaults


@dataclass
class DCFInputs:
    """All inputs required to run a 5-to-10 year Unlevered DCF valuation."""

    ticker: str
    name: str
    base_revenue: float
    current_ebitda_margin: float
    terminal_ebitda_margin: float
    revenue_cagr: float
    da_pct_of_rev: float
    capex_pct_of_rev: float
    nwc_pct_of_rev_change: float
    tax_rate: float
    wacc: float
    terminal_growth_rate: float
    exit_ebitda_multiple: float
    net_debt: float
    minority_interest: float
    current_market_cap: float
    shares_outstanding: float | None = None
    projection_years: int = DCFDefaults.projection_years


@dataclass
class DCFResult:
    """Structured output from the DCF valuation engine."""

    projections: pd.DataFrame
    pv_of_fcff: float
    terminal_value_ggm: float
    pv_of_tv_ggm: float
    implied_ev_ggm: float
    implied_equity_ggm: float
    upside_ggm: float | None
    terminal_value_exit: float
    pv_of_tv_exit: float
    implied_ev_exit: float
    implied_equity_exit: float
    upside_exit: float | None
    implied_share_price_ggm: float | None
    implied_share_price_exit: float | None


def estimate_wacc(
    beta: float | None,
    market_cap: float | None,
    total_debt: float | None,
    risk_free_rate: float = DCFDefaults.risk_free_rate,
    equity_risk_premium: float = DCFDefaults.equity_risk_premium,
    cost_of_debt_pre_tax: float = DCFDefaults.cost_of_debt_pre_tax,
    tax_rate: float = DCFDefaults.effective_tax_rate,
) -> Tuple[float, Dict[str, float]]:
    """Estimate WACC using CAPM for Cost of Equity and market-value capital weights."""
    eff_beta = float(beta) if beta is not None and 0.3 <= beta <= 3.0 else 1.15
    cost_of_equity = risk_free_rate + eff_beta * equity_risk_premium
    cost_of_debt_after_tax = cost_of_debt_pre_tax * (1.0 - tax_rate)

    mcap = max(float(market_cap or 0.0), 1.0)
    debt = max(float(total_debt or 0.0), 0.0)
    total_cap = mcap + debt
    equity_weight = mcap / total_cap
    debt_weight = debt / total_cap

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt_after_tax
    wacc = float(np.clip(wacc, 0.05, 0.18))

    breakdown = {
        "beta_used": eff_beta,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt_after_tax": cost_of_debt_after_tax,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "wacc": wacc,
    }
    return wacc, breakdown


def run_dcf(inputs: DCFInputs) -> DCFResult:
    """Execute Unlevered Free Cash Flow (FCFF) DCF under GGM and Exit Multiple TV."""
    years = list(range(1, inputs.projection_years + 1))
    rows: List[Dict[str, float]] = []

    prev_rev = inputs.base_revenue
    pv_fcff_total = 0.0

    for yr in years:
        # Linearly glide EBITDA margin from current to terminal target
        weight = yr / inputs.projection_years
        ebitda_margin = (
            inputs.current_ebitda_margin * (1.0 - weight)
            + inputs.terminal_ebitda_margin * weight
        )
        # Taper revenue growth slightly in later years if high-growth
        rev = prev_rev * (1.0 + inputs.revenue_cagr)
        delta_rev = rev - prev_rev

        ebitda = rev * ebitda_margin
        da = rev * inputs.da_pct_of_rev
        ebit = ebitda - da
        nopat = ebit * (1.0 - inputs.tax_rate) if ebit > 0 else ebit
        capex = rev * inputs.capex_pct_of_rev
        delta_nwc = delta_rev * inputs.nwc_pct_of_rev_change

        fcff = nopat + da - capex - delta_nwc
        discount_factor = 1.0 / ((1.0 + inputs.wacc) ** yr)
        pv_fcff = fcff * discount_factor
        pv_fcff_total += pv_fcff

        rows.append(
            {
                "year": yr,
                "revenue": rev,
                "ebitda_margin": ebitda_margin,
                "ebitda": ebitda,
                "ebit": ebit,
                "nopat": nopat,
                "da": da,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "fcff": fcff,
                "discount_factor": discount_factor,
                "pv_fcff": pv_fcff,
            }
        )
        prev_rev = rev

    proj_df = pd.DataFrame(rows)
    final_fcff = float(proj_df["fcff"].iloc[-1])
    final_ebitda = float(proj_df["ebitda"].iloc[-1])
    final_df = float(proj_df["discount_factor"].iloc[-1])

    # Terminal Value - Gordon Growth Method
    spread = inputs.wacc - inputs.terminal_growth_rate
    if spread <= 0.005:
        spread = 0.005  # Guard against division by zero / negative spread
    tv_ggm = (final_fcff * (1.0 + inputs.terminal_growth_rate)) / spread
    pv_tv_ggm = tv_ggm * final_df
    implied_ev_ggm = pv_fcff_total + pv_tv_ggm
    implied_equity_ggm = implied_ev_ggm - inputs.net_debt - inputs.minority_interest

    # Terminal Value - Exit Multiple Method
    tv_exit = final_ebitda * inputs.exit_ebitda_multiple
    pv_tv_exit = tv_exit * final_df
    implied_ev_exit = pv_fcff_total + pv_tv_exit
    implied_equity_exit = implied_ev_exit - inputs.net_debt - inputs.minority_interest

    upside_ggm = (
        (implied_equity_ggm / inputs.current_market_cap) - 1.0
        if inputs.current_market_cap and inputs.current_market_cap > 0
        else None
    )
    upside_exit = (
        (implied_equity_exit / inputs.current_market_cap) - 1.0
        if inputs.current_market_cap and inputs.current_market_cap > 0
        else None
    )

    shares = inputs.shares_outstanding
    price_ggm = implied_equity_ggm / shares if shares and shares > 0 else None
    price_exit = implied_equity_exit / shares if shares and shares > 0 else None

    return DCFResult(
        projections=proj_df,
        pv_of_fcff=pv_fcff_total,
        terminal_value_ggm=tv_ggm,
        pv_of_tv_ggm=pv_tv_ggm,
        implied_ev_ggm=implied_ev_ggm,
        implied_equity_ggm=implied_equity_ggm,
        upside_ggm=upside_ggm,
        terminal_value_exit=tv_exit,
        pv_of_tv_exit=pv_tv_exit,
        implied_ev_exit=implied_ev_exit,
        implied_equity_exit=implied_equity_exit,
        upside_exit=upside_exit,
        implied_share_price_ggm=price_ggm,
        implied_share_price_exit=price_exit,
    )


def generate_sensitivity_matrix(
    base_inputs: DCFInputs,
    wacc_steps: List[float] | None = None,
    tgr_steps: List[float] | None = None,
    metric: str = "upside_ggm",
) -> pd.DataFrame:
    """Generate a 2D sensitivity matrix of Implied Equity Value or Upside % across WACC & g."""
    if wacc_steps is None:
        wacc_steps = [
            round(max(0.04, base_inputs.wacc + delta), 4)
            for delta in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)
        ]
    if tgr_steps is None:
        tgr_steps = [0.010, 0.015, 0.020, 0.025, 0.030]

    matrix = pd.DataFrame(
        index=[f"{w * 100:.1f}%" for w in wacc_steps],
        columns=[f"{g * 100:.1f}%" for g in tgr_steps],
        dtype=float,
    )
    matrix.index.name = "WACC"
    matrix.columns.name = "Terminal Growth (g)"

    for w in wacc_steps:
        for g in tgr_steps:
            if w <= g + 0.005:
                matrix.loc[f"{w * 100:.1f}%", f"{g * 100:.1f}%"] = np.nan
                continue
            mod_inputs = DCFInputs(
                ticker=base_inputs.ticker,
                name=base_inputs.name,
                base_revenue=base_inputs.base_revenue,
                current_ebitda_margin=base_inputs.current_ebitda_margin,
                terminal_ebitda_margin=base_inputs.terminal_ebitda_margin,
                revenue_cagr=base_inputs.revenue_cagr,
                da_pct_of_rev=base_inputs.da_pct_of_rev,
                capex_pct_of_rev=base_inputs.capex_pct_of_rev,
                nwc_pct_of_rev_change=base_inputs.nwc_pct_of_rev_change,
                tax_rate=base_inputs.tax_rate,
                wacc=w,
                terminal_growth_rate=g,
                exit_ebitda_multiple=base_inputs.exit_ebitda_multiple,
                net_debt=base_inputs.net_debt,
                minority_interest=base_inputs.minority_interest,
                current_market_cap=base_inputs.current_market_cap,
                shares_outstanding=base_inputs.shares_outstanding,
                projection_years=base_inputs.projection_years,
            )
            res = run_dcf(mod_inputs)
            if metric == "upside_ggm":
                val = (res.upside_ggm * 100.0) if res.upside_ggm is not None else np.nan
            else:
                val = res.implied_equity_ggm / 1e9  # Billions
            matrix.loc[f"{w * 100:.1f}%", f"{g * 100:.1f}%"] = val

    return matrix
