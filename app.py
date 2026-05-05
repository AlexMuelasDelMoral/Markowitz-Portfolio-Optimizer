import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import re

from markowitz import (
    get_price_data,
    calculate_returns_and_cov,
    portfolio_performance,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    efficient_frontier,
    target_return_portfolio,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0E11 !important;
    color: #E8EAED !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #111520 !important;
    border-right: 1px solid #1E2535 !important;
}
[data-testid="stSidebar"] * { color: #C9CDD6 !important; }
[data-testid="stSidebar"] label { font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #606880 !important; }

.app-header { border-bottom: 1px solid #1E2535; padding-bottom: 1.2rem; margin-bottom: 2rem; }
.app-title { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.18em; color: #E8EAED; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
.app-sub { font-size: 0.68rem; color: #606880; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.25rem; }
.section-label {
    font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #606880; border-left: 2px solid #F0B90B;
    padding-left: 0.5rem; margin: 1.5rem 0 0.75rem 0;
}
.status-bar {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #606880; letter-spacing: 0.05em;
    background: #111520; border: 1px solid #1E2535; border-radius: 3px;
    padding: 0.5rem 0.9rem; margin-bottom: 1.8rem;
}
.port-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #606880; margin-bottom: 0.6rem;
}

[data-testid="stMetric"] {
    background: #111520 !important;
    border: 1px solid #1E2535 !important;
    border-radius: 3px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.65rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #606880 !important;
}
[data-testid="stMetricValue"] {
    color: #F0B90B !important; font-size: 1.3rem !important;
    font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

[data-testid="stDataFrame"] {
    background: #111520 !important; border: 1px solid #1E2535 !important;
    border-radius: 3px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stButton"] > button {
    background: #F0B90B !important; color: #0B0E11 !important;
    border: none !important; border-radius: 3px !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; font-size: 0.72rem !important;
    padding: 0.6rem 1.2rem !important;
}
[data-testid="stButton"] > button:hover { background: #d4a30a !important; }

[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
    background: #0d1017 !important; border: 1px solid #1E2535 !important;
    color: #E8EAED !important; border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important;
}
[data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {
    border-color: #F0B90B !important; box-shadow: 0 0 0 1px #F0B90B30 !important;
}
[data-testid="stRadio"] label { font-size: 0.78rem !important; color: #C9CDD6 !important; }
[data-testid="stNumberInput"] input {
    background: #0d1017 !important; border: 1px solid #1E2535 !important;
    color: #E8EAED !important; font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSlider"] { margin-top: 0.3rem; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: #F0B90B !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #606880 !important; font-size: 0.68rem !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Portfolio Optimizer</div>
    <div class="app-sub">Markowitz Mean-Variance Framework &mdash; Modern Portfolio Theory</div>
</div>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
AMBER   = "#F0B90B"
CYAN    = "#00C8FF"
RED_NEG = "#EF5350"
GREEN   = "#26A69A"
ASSET_PALETTE = [
    "#F0B90B", "#00C8FF", "#FF6B9D", "#A78BFA",
    "#34D399", "#F87171", "#60A5FA", "#FBBF24",
    "#818CF8", "#6EE7B7",
]

PLOTLY_BASE = dict(
    paper_bgcolor="#0B0E11",
    plot_bgcolor="#0B0E11",
    font=dict(family="'JetBrains Mono', monospace", color="#C9CDD6", size=11),
    xaxis=dict(gridcolor="#1A1F2E", linecolor="#1E2535", zerolinecolor="#1E2535", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1A1F2E", linecolor="#1E2535", zerolinecolor="#1E2535", tickfont=dict(size=10)),
    margin=dict(l=55, r=25, t=35, b=50),
    hovermode="closest",
    legend=dict(
        bgcolor="#111520", bordercolor="#1E2535", borderwidth=1,
        font=dict(size=10), orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0
    ),
)


# ── Ticker parser ─────────────────────────────────────────────────────────────
def parse_tickers(raw: str) -> list:
    """
    Extract ticker symbols from free-form text.
    Handles: comma-separated, newline-separated, tab-separated (Excel paste),
    and rows with weights/numbers mixed in.
    """
    tickers = []
    for line in raw.splitlines():
        parts = re.split(r"[,\t;| ]+", line)
        for part in parts:
            token = part.strip().upper()
            # valid US/intl ticker: 1-5 letters, optionally .X or -X suffix
            if re.match(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$", token):
                tickers.append(token)
    seen = set()
    return [t for t in tickers if not (t in seen or seen.add(t))]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label" style="margin-top:0.2rem">Portfolio Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "mode",
        ["Type tickers", "Paste from spreadsheet"],
        label_visibility="collapsed",
    )

    if input_mode == "Type tickers":
        tickers_raw = st.text_input(
            "Tickers",
            value="AAPL, MSFT, GOOGL, AMZN, NVDA",
            placeholder="AAPL, MSFT, GOOGL ...",
            label_visibility="collapsed",
        )
    else:
        st.caption("Paste from Excel, Bloomberg, or any source. Tickers are extracted automatically — weights and numbers are ignored.")
        tickers_raw = st.text_area(
            "Paste portfolio",
            height=155,
            placeholder="AAPL\t0.25\nMSFT\t0.30\nGOOGL\t0.20\n...",
            label_visibility="collapsed",
        )

    st.markdown('<div class="section-label">Date Range</div>', unsafe_allow_html=True)
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", value=date.today() - timedelta(days=5 * 365))
    with col_e:
        end_date = st.date_input("To", value=date.today())

    st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        value=0.04, min_value=0.0, max_value=0.20, step=0.005, format="%.3f",
    )

    run_button = st.button("Run Optimization", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if run_button:
    tickers = parse_tickers(tickers_raw)

    if len(tickers) < 2:
        st.error("Enter at least 2 valid ticker symbols.")
        st.stop()

    with st.spinner("Fetching market data..."):
        try:
            prices = get_price_data(tickers, start_date, end_date)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    if prices.empty:
        st.error("No price data returned. Check your tickers or date range.")
        st.stop()

    mean_returns, cov_matrix = calculate_returns_and_cov(prices)
    daily_returns = prices.pct_change().dropna()

    # ── Status bar ──
    st.markdown(
        f'<div class="status-bar">'
        f'LOADED &nbsp;|&nbsp; {len(prices.columns)} assets &nbsp;|&nbsp; '
        f'{len(prices):,} trading days &nbsp;|&nbsp; '
        f'{prices.index[0].strftime("%d %b %Y")} &rarr; {prices.index[-1].strftime("%d %b %Y")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 1. Normalized price chart ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Price Performance — Indexed to 100</div>', unsafe_allow_html=True)
    normed = prices / prices.iloc[0] * 100

    fig_price = go.Figure()
    for i, col in enumerate(normed.columns):
        fig_price.add_trace(go.Scatter(
            x=normed.index, y=normed[col], name=col, mode="lines",
            line=dict(width=1.5, color=ASSET_PALETTE[i % len(ASSET_PALETTE)]),
            hovertemplate=f"<b>{col}</b>: %{{y:.1f}}<extra></extra>",
        ))
    fig_price.update_layout(**PLOTLY_BASE, height=300)
    fig_price.update_layout(xaxis_title="", yaxis_title="Index (100 = start)")
    st.plotly_chart(fig_price, use_container_width=True)

    # ── 2. Asset statistics table ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Asset Statistics</div>', unsafe_allow_html=True)
    vols = [np.sqrt(cov_matrix.iloc[i, i]) for i in range(len(mean_returns))]
    sharpes = [(mean_returns.iloc[i] - risk_free_rate) / vols[i] for i in range(len(mean_returns))]
    stats_df = pd.DataFrame({
        "Ticker":          mean_returns.index,
        "Ann. Return":     [f"{r:.2%}" for r in mean_returns],
        "Ann. Volatility": [f"{v:.2%}" for v in vols],
        "Sharpe Ratio":    [f"{s:.2f}" for s in sharpes],
        "Max Drawdown":    [
            f"{((daily_returns[t].add(1).cumprod() / daily_returns[t].add(1).cumprod().cummax()) - 1).min():.2%}"
            for t in daily_returns.columns
        ],
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # ── 3. Optimal portfolios ─────────────────────────────────────────────────
    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
    min_vol_w    = min_volatility_portfolio(mean_returns, cov_matrix)
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix, risk_free_rate)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(min_vol_w,    mean_returns, cov_matrix, risk_free_rate)

    st.markdown('<div class="section-label">Optimal Portfolios</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    def render_portfolio(container, label, weights, ret, vol, sharpe, accent):
        with container:
            st.markdown(f'<div class="port-label">{label}</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Return",     f"{ret:.2%}")
            m2.metric("Volatility", f"{vol:.2%}")
            m3.metric("Sharpe",     f"{sharpe:.2f}")

            wdf = (
                pd.DataFrame({"Ticker": mean_returns.index, "Weight": weights})
                .sort_values("Weight", ascending=False)
            )
            fig_w = go.Figure(go.Bar(
                x=wdf["Ticker"], y=wdf["Weight"],
                marker_color=accent, marker_line_width=0,
                text=[f"{w:.1%}" for w in wdf["Weight"]],
                textposition="outside",
                textfont=dict(size=9, color="#606880"),
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            ))
            fig_w.update_layout(
                **PLOTLY_BASE, height=210, showlegend=False,
                xaxis_title="", yaxis_title="",
                margin=dict(l=10, r=10, t=10, b=30),
            )
            fig_w.update_yaxes(tickformat=".0%", showgrid=False)
            st.plotly_chart(fig_w, use_container_width=True)

    render_portfolio(col1, "Max Sharpe Ratio",  max_sharpe_w, ms_ret, ms_vol, ms_sharpe, AMBER)
    render_portfolio(col2, "Min Volatility",    min_vol_w,    mv_ret, mv_vol, mv_sharpe, CYAN)

    # ── 4. Efficient frontier ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Efficient Frontier</div>', unsafe_allow_html=True)
    with st.spinner("Computing efficient frontier..."):
        frontier = efficient_frontier(mean_returns, cov_matrix, num_points=60)

    f_vols = [p[1] for p in frontier]
    f_rets = [p[0] for p in frontier]

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=f_vols, y=f_rets, mode="lines", name="Efficient Frontier",
        line=dict(color="#2A3550", width=2.5),
    ))
    for i, ticker in enumerate(mean_returns.index):
        a_vol = np.sqrt(cov_matrix.iloc[i, i])
        a_ret = mean_returns.iloc[i]
        fig_ef.add_trace(go.Scatter(
            x=[a_vol], y=[a_ret], mode="markers+text", name=ticker,
            text=[ticker], textposition="top center",
            marker=dict(size=7, color="#3A4560"),
            textfont=dict(size=9, color="#606880"),
            showlegend=False,
            hovertemplate=f"<b>{ticker}</b><br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
        ))
    fig_ef.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret], mode="markers", name="Max Sharpe",
        marker=dict(color=AMBER, size=13, symbol="diamond",
                    line=dict(color="#0B0E11", width=1.5)),
        hovertemplate=f"<b>Max Sharpe</b><br>Vol: {ms_vol:.2%}<br>Return: {ms_ret:.2%}<br>Sharpe: {ms_sharpe:.2f}<extra></extra>",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret], mode="markers", name="Min Volatility",
        marker=dict(color=CYAN, size=13, symbol="diamond",
                    line=dict(color="#0B0E11", width=1.5)),
        hovertemplate=f"<b>Min Volatility</b><br>Vol: {mv_vol:.2%}<br>Return: {mv_ret:.2%}<br>Sharpe: {mv_sharpe:.2f}<extra></extra>",
    ))
    fig_ef.update_layout(
        **PLOTLY_BASE, height=460,
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
    )
    fig_ef.update_xaxes(tickformat=".1%")
    fig_ef.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── 5. Correlation matrix ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = daily_returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#0d1017"], [0.5, "#1A2540"], [1.0, AMBER]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=10, color="#C9CDD6"),
        hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color="#606880", size=9),
            outlinewidth=0, thickness=10,
        ),
    ))
    fig_corr.update_layout(**PLOTLY_BASE, height=320, margin=dict(l=70, r=30, t=20, b=70))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── 6. Custom target return ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Custom Target Return</div>', unsafe_allow_html=True)
    target = st.slider(
        "Target return",
        min_value=float(mean_returns.min()),
        max_value=float(mean_returns.max()),
        value=float(ms_ret),
        step=0.005,
        format="%.3f",
        label_visibility="collapsed",
    )
    custom_w = target_return_portfolio(mean_returns, cov_matrix, target)
    if custom_w is not None:
        c_ret, c_vol, c_sharpe = portfolio_performance(custom_w, mean_returns, cov_matrix, risk_free_rate)
        ca, cb, cc = st.columns(3)
        ca.metric("Return",     f"{c_ret:.2%}")
        cb.metric("Volatility", f"{c_vol:.2%}")
        cc.metric("Sharpe",     f"{c_sharpe:.2f}")
        cdf = (
            pd.DataFrame({"Ticker": mean_returns.index, "Weight": [f"{w:.2%}" for w in custom_w]})
            .assign(_w=custom_w)
            .sort_values("_w", ascending=False)
            .drop(columns="_w")
        )
        st.dataframe(cdf, hide_index=True, use_container_width=True)
    else:
        st.warning("No feasible portfolio found for this target return.")

else:
    st.markdown("""
    <div style="margin:6rem auto;text-align:center;max-width:420px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:2.5rem;
                    letter-spacing:0.15em;color:#1E2535;margin-bottom:1rem;">MPT</div>
        <div style="font-size:0.68rem;color:#2A3550;letter-spacing:0.18em;text-transform:uppercase;">
            Configure inputs in the sidebar and run optimization
        </div>
    </div>
    """, unsafe_allow_html=True)