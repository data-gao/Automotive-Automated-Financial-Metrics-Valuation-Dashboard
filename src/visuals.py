"""Institutional-grade Plotly visualization generators for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dcf_engine import DCFResult

REGION_COLORS = {
    "Europe": "#1D4ED8",          # Deep Cobalt Blue
    "China": "#DC2626",           # Crimson Red
    "Global Benchmark": "#D97706",  # Amber Gold
}


def plot_profitability_vs_rd(df: pd.DataFrame, currency: str) -> go.Figure:
    """Bubble chart: EBITDA Margin (%) vs. R&D Intensity (%), sized by Market Cap."""
    plot_df = df.dropna(subset=["ebitda_margin", "rd_intensity"]).copy()
    plot_df["ebitda_margin_pct"] = plot_df["ebitda_margin"] * 100
    plot_df["rd_intensity_pct"] = plot_df["rd_intensity"] * 100
    plot_df["size_mcap"] = plot_df["market_cap_bn"].fillna(10.0).clip(lower=5.0)

    fig = px.scatter(
        plot_df,
        x="rd_intensity_pct",
        y="ebitda_margin_pct",
        size="size_mcap",
        color="region",
        color_discrete_map=REGION_COLORS,
        hover_name="name",
        text="ticker",
        size_max=48,
        hover_data={
            "region": True,
            "market_cap_bn": ":.1f",
            "ebitda_margin_pct": ":.1f",
            "rd_intensity_pct": ":.1f",
            "size_mcap": False,
        },
        labels={
            "rd_intensity_pct": "R&D Intensity (% of Revenue)",
            "ebitda_margin_pct": "EBITDA Margin (%)",
            "market_cap_bn": f"Market Cap ({currency} Bn)",
            "region": "Region",
        },
        title="R&D Commitment vs. Operating Profitability (Bubble = Market Cap)",
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=480, template="plotly_white", legend_title_text="Region")
    return fig


def plot_valuation_multiples(
    df: pd.DataFrame, multiple_col: str, title: str, yaxis_label: str
) -> go.Figure:
    """Bar chart comparing relative valuation multiples across peers."""
    plot_df = df.dropna(subset=[multiple_col]).sort_values(multiple_col, ascending=False).copy()
    # Cap extreme outliers for visual readability while keeping true value in hover
    fig = px.bar(
        plot_df,
        x="name",
        y=multiple_col,
        color="region",
        color_discrete_map=REGION_COLORS,
        text=plot_df[multiple_col].map(lambda x: f"{x:.1f}x"),
        title=title,
        labels={"name": "", multiple_col: yaxis_label, "region": "Region"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=430,
        template="plotly_white",
        xaxis_tickangle=-25,
        showlegend=True,
    )
    return fig


def plot_roic_vs_leverage(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: ROIC (%) vs. Net Debt / EBITDA (x)."""
    plot_df = df.dropna(subset=["roic", "net_debt_to_ebitda"]).copy()
    plot_df["roic_pct"] = plot_df["roic"] * 100

    fig = px.scatter(
        plot_df,
        x="net_debt_to_ebitda",
        y="roic_pct",
        color="region",
        color_discrete_map=REGION_COLORS,
        text="ticker",
        hover_name="name",
        title="Capital Efficiency (ROIC) vs. Balance Sheet Leverage (Net Debt / EBITDA)",
        labels={
            "net_debt_to_ebitda": "Net Debt / EBITDA (x)",
            "roic_pct": "ROIC (%)",
            "region": "Region",
        },
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")), textposition="top center")
    fig.add_hline(y=8.0, line_dash="dot", line_color="green", annotation_text="~8% WACC Hurdle")
    fig.add_vline(x=0.0, line_dash="dash", line_color="gray", annotation_text="Net Cash")
    fig.update_layout(height=460, template="plotly_white")
    return fig


def plot_historical_trends(
    hist_df: pd.DataFrame, metric_col: str, title: str, y_title: str, is_pct: bool = False
) -> go.Figure:
    """Multi-year trend line chart across selected peers."""
    plot_df = hist_df.dropna(subset=[metric_col]).copy()
    if is_pct:
        plot_df["plot_val"] = plot_df[metric_col] * 100
    else:
        plot_df["plot_val"] = plot_df[metric_col]

    fig = px.line(
        plot_df,
        x="fiscal_year",
        y="plot_val",
        color="name",
        markers=True,
        title=title,
        labels={"fiscal_year": "Fiscal Year", "plot_val": y_title, "name": "Company"},
    )
    fig.update_layout(height=430, template="plotly_white", hovermode="x unified")
    return fig


def plot_dcf_sensitivity_heatmap(
    matrix: pd.DataFrame,
    title: str = "DCF Implied Upside / (Downside) Sensitivity (%)",
    is_pct: bool = True,
) -> go.Figure:
    """Render an annotated 2D sensitivity heatmap (WACC vs. Terminal Growth Rate)."""
    z_vals = matrix.values.astype(float)
    text_vals = np.empty(z_vals.shape, dtype=object)
    for i in range(z_vals.shape[0]):
        for j in range(z_vals.shape[1]):
            v = z_vals[i, j]
            if np.isnan(v):
                text_vals[i, j] = "N/A"
            elif is_pct:
                text_vals[i, j] = f"{v:+.1f}%"
            else:
                text_vals[i, j] = f"{v:,.1f}B"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="RdYlGn",
            zmid=0.0 if is_pct else None,
            colorbar=dict(title="Upside %" if is_pct else "Value (Bn)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Terminal Growth Rate (g)",
        yaxis_title="WACC",
        yaxis_autorange="reversed",
        height=440,
        template="plotly_white",
    )
    return fig


def plot_dcf_valuation_bridge(
    dcf_res: DCFResult, current_mcap: float, net_debt: float, currency: str
) -> go.Figure:
    """Compare Implied Enterprise/Equity Value (GGM & Exit Multiple) vs Current Market Cap."""
    categories = [
        "PV of 5Y FCFF",
        "PV of TV (GGM)",
        "Implied EV (GGM)",
        "Implied Equity (GGM)",
        "Implied Equity (Exit)",
        "Current Market Cap",
    ]
    values = [
        dcf_res.pv_of_fcff / 1e9,
        dcf_res.pv_of_tv_ggm / 1e9,
        dcf_res.implied_ev_ggm / 1e9,
        dcf_res.implied_equity_ggm / 1e9,
        dcf_res.implied_equity_exit / 1e9,
        (current_mcap or 0.0) / 1e9,
    ]
    colors = ["#64748B", "#94A3B8", "#3B82F6", "#1D4ED8", "#0F766E", "#D97706"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:,.1f}B" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"DCF Valuation Summary vs. Current Market Cap ({currency} Bn)",
        yaxis_title=f"Value ({currency} Billions)",
        height=420,
        template="plotly_white",
    )
    return fig
