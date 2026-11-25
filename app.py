"""Streamlit Web Dashboard for Sino-European Automotive Valuation & DCF Analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    FALLBACK_FX_TO_EUR,
    PROCESSED_DATA_DIR,
    SUPPORTED_CURRENCIES,
    DCFDefaults,
)
from src.dcf_engine import (
    DCFInputs,
    estimate_wacc,
    generate_sensitivity_matrix,
    run_dcf,
)
from src.ingestion import run_ingestion
from src.metrics import compute_historical_metrics, compute_peer_metrics
from src.visuals import (
    plot_dcf_sensitivity_heatmap,
    plot_dcf_valuation_bridge,
    plot_historical_trends,
    plot_profitability_vs_rd,
    plot_roic_vs_leverage,
    plot_valuation_multiples,
)

st.set_page_config(
    page_title="Sino-European Automotive Valuation Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load processed parquet/csv datasets and metadata from disk."""
    summary_path = PROCESSED_DATA_DIR / "peers_summary.parquet"
    hist_path = PROCESSED_DATA_DIR / "historical_financials.parquet"
    meta_path = PROCESSED_DATA_DIR / "metadata.json"

    if not summary_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}

    summary_df = pd.read_parquet(summary_path)
    hist_df = pd.read_parquet(hist_path) if hist_path.exists() else pd.DataFrame()
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    return summary_df, hist_df, meta


def main() -> None:
    st.title("🚗 Sino-European Automotive Peer Group & Valuation Dashboard")
    st.caption(
        "Automated Financial Screener, Cross-Currency Comps & DCF Sensitivity Engine "
        "| Built for M&A, Equity Research & Corporate Strategy"
    )

    summary_raw, hist_raw, meta = load_cached_data()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")

        target_ccy = st.selectbox(
            "Display Currency",
            options=[c for c in SUPPORTED_CURRENCIES if c != "HKD"],
            index=0,
            help="All financial statements & market caps are dynamically normalized via spot FX.",
        )

        peer_universe = st.radio(
            "Peer Universe",
            options=["OEMs Only", "Tier-1 Suppliers", "Combined (OEM + Tier-1)"],
            index=0,
            help="Switch between Automotive OEMs, Tier-1 Suppliers, or a combined value-chain view.",
        )

        st.divider()
        st.subheader("🔄 Data Pipeline")
        last_updated = meta.get("last_updated_utc", "Not yet ingested")
        st.caption(f"Last Updated (UTC): `{last_updated[:19]}`")

        if st.button("Refresh Live Market Data (yfinance)", use_container_width=True):
            with st.spinner("Fetching live financial statements & spot FX from Yahoo Finance..."):
                run_ingestion(include_disabled=True)
                st.cache_data.clear()
                st.rerun()

    if summary_raw.empty:
        st.warning(
            "No cached financial data found in `data/processed/`. "
            "Click **Refresh Live Market Data (yfinance)** in the sidebar or run "
            "`python -m src.ingestion` to populate the dataset."
        )
        return

    fx_to_eur = meta.get("fx_to_eur", FALLBACK_FX_TO_EUR)
    df = compute_peer_metrics(summary_raw, target_currency=target_ccy, fx_to_eur=fx_to_eur)
    hist_df = compute_historical_metrics(hist_raw, target_currency=target_ccy, fx_to_eur=fx_to_eur)

    # Filter by category (OEM vs Tier-1 vs Combined) & region
    if "category" in df.columns:
        if peer_universe == "OEMs Only":
            df = df[df["category"] == "OEM"].copy()
            if not hist_df.empty and "category" in hist_df.columns:
                hist_df = hist_df[hist_df["category"] == "OEM"].copy()
        elif peer_universe == "Tier-1 Suppliers":
            df = df[df["category"] == "Tier-1"].copy()
            if not hist_df.empty and "category" in hist_df.columns:
                hist_df = hist_df[hist_df["category"] == "Tier-1"].copy()

    if df.empty:
        st.info(
            f"No data available for **{peer_universe}** in the current cache. "
            "Click **Refresh Live Market Data (yfinance)** in the sidebar to ingest all peers."
        )
        return

    with st.sidebar:
        all_regions = sorted(df["region"].dropna().unique().tolist())
        selected_regions = st.multiselect("Filter Regions", all_regions, default=all_regions)
        df = df[df["region"].isin(selected_regions)].copy()
        if not hist_df.empty:
            hist_df = hist_df[hist_df["ticker"].isin(df["ticker"])].copy()

    # Executive KPI Strip
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peers Tracked", f"{len(df)}")
    med_ebitda_m = df["ebitda_margin"].median() * 100 if not df["ebitda_margin"].dropna().empty else 0.0
    col2.metric("Median EBITDA Margin", f"{med_ebitda_m:.1f}%")
    med_rd = df["rd_intensity"].median() * 100 if not df["rd_intensity"].dropna().empty else 0.0
    col3.metric("Median R&D / Sales", f"{med_rd:.1f}%")
    med_ev_ebitda = df["ev_to_ebitda"].median() if not df["ev_to_ebitda"].dropna().empty else 0.0
    col4.metric("Median EV / EBITDA", f"{med_ev_ebitda:.1f}x")

    # Main Tabs
    tab_comps, tab_trends, tab_dcf, tab_data = st.tabs(
        [
            "📊 Trading Comps & Ratios",
            "📈 Historical Trajectories",
            "🧮 DCF & Sensitivity Engine",
            "📑 Raw Normalized Data",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Trading Comps & Operational Ratios
    # ------------------------------------------------------------------
    with tab_comps:
        st.subheader("1. Institutional Trading Comps Table")
        comps_cols = {
            "ticker": "Ticker",
            "name": "Company",
            "category": "Type",
            "region": "Region",
            "market_cap_bn": f"MCap ({target_ccy} Bn)",
            "enterprise_value_bn": f"EV ({target_ccy} Bn)",
            "revenue_bn": f"Revenue ({target_ccy} Bn)",
            "ebitda_margin": "EBITDA %",
            "ebit_margin": "EBIT %",
            "rd_intensity": "R&D / Sales",
            "roic": "ROIC %",
            "net_debt_to_ebitda": "Net Debt/EBITDA",
            "ev_to_ebitda": "EV/EBITDA",
            "ev_to_sales": "EV/Sales",
            "forward_pe": "Fwd P/E",
        }
        available_cols = [c for c in comps_cols if c in df.columns]
        display_df = df[available_cols].rename(columns=comps_cols)

        st.dataframe(
            display_df.style.format(
                {
                    f"MCap ({target_ccy} Bn)": "{:,.1f}",
                    f"EV ({target_ccy} Bn)": "{:,.1f}",
                    f"Revenue ({target_ccy} Bn)": "{:,.1f}",
                    "EBITDA %": "{:.1%}",
                    "EBIT %": "{:.1%}",
                    "R&D / Sales": "{:.1%}",
                    "ROIC %": "{:.1%}",
                    "Net Debt/EBITDA": "{:.2f}x",
                    "EV/EBITDA": "{:.1f}x",
                    "EV/Sales": "{:.2f}x",
                    "Fwd P/E": "{:.1f}x",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_profitability_vs_rd(df, target_ccy),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plot_roic_vs_leverage(df),
                use_container_width=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                plot_valuation_multiples(
                    df, "ev_to_ebitda", "Enterprise Value / EBITDA (x)", "EV / EBITDA (x)"
                ),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                plot_valuation_multiples(
                    df, "ev_to_sales", "Enterprise Value / Sales (x)", "EV / Sales (x)"
                ),
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # TAB 2: Multi-Year Historical Trajectories
    # ------------------------------------------------------------------
    with tab_trends:
        if hist_df.empty:
            st.info("No historical statement data available.")
        else:
            st.subheader("Multi-Year Financial Performance Comparison")
            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(
                    plot_historical_trends(
                        hist_df,
                        "revenue_bn",
                        f"Annual Revenue Trajectory ({target_ccy} Bn)",
                        f"Revenue ({target_ccy} Bn)",
                    ),
                    use_container_width=True,
                )
            with h2:
                st.plotly_chart(
                    plot_historical_trends(
                        hist_df,
                        "ebitda_margin",
                        "EBITDA Margin Evolution (%)",
                        "EBITDA Margin (%)",
                        is_pct=True,
                    ),
                    use_container_width=True,
                )

            h3, h4 = st.columns(2)
            with h3:
                st.plotly_chart(
                    plot_historical_trends(
                        hist_df,
                        "rd_intensity",
                        "R&D Intensity (% of Revenue)",
                        "R&D / Revenue (%)",
                        is_pct=True,
                    ),
                    use_container_width=True,
                )
            with h4:
                st.plotly_chart(
                    plot_historical_trends(
                        hist_df,
                        "fcf_bn",
                        f"Free Cash Flow Generation ({target_ccy} Bn)",
                        f"FCF ({target_ccy} Bn)",
                    ),
                    use_container_width=True,
                )

    # ------------------------------------------------------------------
    # TAB 3: Interactive DCF & Sensitivity Engine
    # ------------------------------------------------------------------
    with tab_dcf:
        st.subheader("Unlevered DCF Model & 2D Sensitivity Analysis")
        peer_options = df["name"].tolist()
        selected_name = st.selectbox("Select Target Company for DCF Valuation", peer_options)
        row = df[df["name"] == selected_name].iloc[0]

        base_rev = float(row["revenue_ccy"] or 50e9)
        cur_ebitda_m = float(row["ebitda_margin"] if pd.notna(row["ebitda_margin"]) else 0.10)
        da_pct = float(
            (row["da_ccy"] / base_rev)
            if pd.notna(row.get("da_ccy")) and base_rev > 0
            else 0.05
        )
        capex_pct = float(
            (row["capex_ccy"] / base_rev)
            if pd.notna(row.get("capex_ccy")) and base_rev > 0
            else 0.06
        )
        est_wacc, wacc_info = estimate_wacc(
            beta=row.get("beta"),
            market_cap=row.get("market_cap_ccy"),
            total_debt=row.get("total_debt_ccy"),
        )

        with st.expander("🔧 Adjust DCF Model Assumptions", expanded=True):
            a1, a2, a3, a4 = st.columns(4)
            rev_cagr = a1.slider("5Y Revenue CAGR (%)", -5.0, 30.0, 6.0, 0.5) / 100.0
            term_ebitda_m = (
                a2.slider(
                    "Terminal EBITDA Margin (%)",
                    2.0,
                    25.0,
                    float(round(max(0.04, min(0.22, cur_ebitda_m)) * 100, 1)),
                    0.5,
                )
                / 100.0
            )
            wacc_val = (
                a3.slider("WACC (%)", 5.0, 16.0, float(round(est_wacc * 100, 1)), 0.25) / 100.0
            )
            tgr_val = a4.slider("Terminal Growth Rate g (%)", 0.5, 4.0, 2.0, 0.25) / 100.0

            b1, b2, b3, b4 = st.columns(4)
            capex_slider = (
                b1.slider("CapEx % of Revenue", 2.0, 15.0, float(round(min(0.14, max(0.03, capex_pct)) * 100, 1)), 0.5)
                / 100.0
            )
            da_slider = (
                b2.slider("D&A % of Revenue", 2.0, 12.0, float(round(min(0.11, max(0.03, da_pct)) * 100, 1)), 0.5)
                / 100.0
            )
            exit_mult = b3.slider("Exit EV/EBITDA Multiple (x)", 3.0, 25.0, DCFDefaults.exit_ebitda_multiple, 0.5)
            tax_slider = b4.slider("Effective Tax Rate (%)", 15.0, 35.0, 25.0, 1.0) / 100.0

        dcf_inputs = DCFInputs(
            ticker=str(row["ticker"]),
            name=str(row["name"]),
            base_revenue=base_rev,
            current_ebitda_margin=cur_ebitda_m,
            terminal_ebitda_margin=term_ebitda_m,
            revenue_cagr=rev_cagr,
            da_pct_of_rev=da_slider,
            capex_pct_of_rev=capex_slider,
            nwc_pct_of_rev_change=0.02,
            tax_rate=tax_slider,
            wacc=wacc_val,
            terminal_growth_rate=tgr_val,
            exit_ebitda_multiple=exit_mult,
            net_debt=float(row["net_debt_ccy"] or 0.0),
            minority_interest=float(row.get("minority_interest_ccy") or 0.0),
            current_market_cap=float(row["market_cap_ccy"] or 0.0),
            shares_outstanding=float(row["shares_outstanding"]) if pd.notna(row.get("shares_outstanding")) else None,
        )
        dcf_res = run_dcf(dcf_inputs)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            f"Implied EV (GGM)",
            f"{dcf_res.implied_ev_ggm / 1e9:,.1f}B {target_ccy}",
        )
        k2.metric(
            f"Implied Equity (GGM)",
            f"{dcf_res.implied_equity_ggm / 1e9:,.1f}B {target_ccy}",
            delta=f"{dcf_res.upside_ggm * 100:+.1f}% vs MCap" if dcf_res.upside_ggm is not None else None,
        )
        k3.metric(
            f"Implied Equity (Exit {exit_mult:.1f}x)",
            f"{dcf_res.implied_equity_exit / 1e9:,.1f}B {target_ccy}",
            delta=f"{dcf_res.upside_exit * 100:+.1f}% vs MCap" if dcf_res.upside_exit is not None else None,
        )
        k4.metric("CAPM Estimated WACC", f"{wacc_info['wacc'] * 100:.2f}%", f"Beta: {wacc_info['beta_used']:.2f}")

        d1, d2 = st.columns(2)
        with d1:
            sens_matrix = generate_sensitivity_matrix(dcf_inputs, metric="upside_ggm")
            st.plotly_chart(
                plot_dcf_sensitivity_heatmap(
                    sens_matrix,
                    title=f"{row['ticker']} — DCF Implied Upside/(Downside) % (WACC vs. g)",
                    is_pct=True,
                ),
                use_container_width=True,
            )
        with d2:
            st.plotly_chart(
                plot_dcf_valuation_bridge(
                    dcf_res,
                    current_mcap=dcf_inputs.current_market_cap,
                    net_debt=dcf_inputs.net_debt,
                    currency=target_ccy,
                ),
                use_container_width=True,
            )

        with st.expander("📋 5-Year Unlevered Free Cash Flow (FCFF) Projection Table"):
            proj_disp = dcf_res.projections.copy()
            for col in ["revenue", "ebitda", "ebit", "nopat", "da", "capex", "delta_nwc", "fcff", "pv_fcff"]:
                proj_disp[col] = proj_disp[col] / 1e9
            proj_disp["ebitda_margin"] = proj_disp["ebitda_margin"] * 100
            st.dataframe(
                proj_disp.style.format(
                    {
                        "revenue": "{:,.2f}B",
                        "ebitda_margin": "{:.1f}%",
                        "ebitda": "{:,.2f}B",
                        "ebit": "{:,.2f}B",
                        "nopat": "{:,.2f}B",
                        "da": "{:,.2f}B",
                        "capex": "{:,.2f}B",
                        "delta_nwc": "{:,.2f}B",
                        "fcff": "{:,.2f}B",
                        "discount_factor": "{:.3f}",
                        "pv_fcff": "{:,.2f}B",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ------------------------------------------------------------------
    # TAB 4: Raw Normalized Data & FX Matrix
    # ------------------------------------------------------------------
    with tab_data:
        st.subheader(f"Normalized Dataset ({target_ccy}) & Spot FX Snapshot")
        st.json({"fx_to_eur": fx_to_eur, "peer_count": len(df)})
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
