"""Unit tests for currency normalization, financial ratios, and DCF sensitivity engine."""

import pandas as pd
import pytest

from src.dcf_engine import DCFInputs, estimate_wacc, generate_sensitivity_matrix, run_dcf
from src.metrics import compute_peer_metrics
from src.normalizer import convert_amount


def test_currency_conversion_via_eur_pivot():
    fx = {"EUR": 1.0, "USD": 0.90, "CNY": 0.125, "HKD": 0.115}
    # 100 USD -> EUR = 90 EUR
    assert convert_amount(100.0, "USD", "EUR", fx) == pytest.approx(90.0)
    # 100 EUR -> EUR = 100 EUR
    assert convert_amount(100.0, "EUR", "EUR", fx) == pytest.approx(100.0)
    # 100 CNY -> USD = (100 * 0.125) / 0.90
    assert convert_amount(100.0, "CNY", "USD", fx) == pytest.approx(12.5 / 0.90)


def test_compute_peer_metrics_ratios():
    sample = pd.DataFrame(
        [
            {
                "ticker": "BMW.DE",
                "name": "BMW Group",
                "region": "Europe",
                "market_cap_eur": 60e9,
                "enterprise_value_eur": 90e9,
                "revenue_eur": 150e9,
                "gross_profit_eur": 30e9,
                "ebitda_eur": 24e9,
                "ebit_eur": 18e9,
                "net_income_eur": 12e9,
                "rd_expense_eur": 7.5e9,
                "da_eur": 6e9,
                "interest_expense_eur": 1.0e9,
                "tax_provision_eur": 4.5e9,
                "pretax_income_eur": 18e9,
                "total_debt_eur": 50e9,
                "cash_eur": 20e9,
                "total_equity_eur": 90e9,
                "minority_interest_eur": 0.0,
                "ocf_eur": 18e9,
                "capex_eur": 8e9,
                "fcf_eur": 10e9,
            }
        ]
    )
    res = compute_peer_metrics(sample, target_currency="EUR", fx_to_eur={"EUR": 1.0})
    assert res.loc[0, "ebitda_margin"] == pytest.approx(0.16)
    assert res.loc[0, "rd_intensity"] == pytest.approx(0.05)
    assert res.loc[0, "net_debt_bn"] == pytest.approx(30.0)
    assert res.loc[0, "ev_to_ebitda"] == pytest.approx(90e9 / 24e9)
    assert res.loc[0, "ev_to_sales"] == pytest.approx(90e9 / 150e9)


def test_dcf_engine_and_sensitivity_matrix():
    wacc, _ = estimate_wacc(beta=1.1, market_cap=60e9, total_debt=20e9)
    assert 0.05 <= wacc <= 0.15

    inputs = DCFInputs(
        ticker="TEST",
        name="Test OEM",
        base_revenue=100e9,
        current_ebitda_margin=0.12,
        terminal_ebitda_margin=0.14,
        revenue_cagr=0.05,
        da_pct_of_rev=0.04,
        capex_pct_of_rev=0.05,
        nwc_pct_of_rev_change=0.02,
        tax_rate=0.25,
        wacc=0.09,
        terminal_growth_rate=0.02,
        exit_ebitda_multiple=6.0,
        net_debt=10e9,
        minority_interest=0.0,
        current_market_cap=50e9,
        shares_outstanding=1e9,
    )
    res = run_dcf(inputs)
    assert len(res.projections) == 5
    assert res.implied_ev_ggm > res.pv_of_fcff
    assert res.implied_equity_ggm == pytest.approx(res.implied_ev_ggm - 10e9)

    matrix = generate_sensitivity_matrix(inputs)
    assert matrix.shape == (7, 5)
    # Higher WACC should yield lower upside than lower WACC at the same terminal growth rate
    assert matrix.iloc[0, 2] > matrix.iloc[-1, 2]
