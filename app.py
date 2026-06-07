import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_pipeline import build_dataset, walk_forward_splits, get_feature_cols
from model import EnsemblePredictor
from backtest import Backtester

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSage · Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — Warm ivory · Deep navy · Saffron accent
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:opsz,wght@9..40,300;9..40,500;9..40,700&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif !important;
  color: #1a1a2e !important;
}
.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 8% 0%,  rgba(245,158,11,.09) 0%, transparent 55%),
    radial-gradient(ellipse 60% 40% at 92% 100%, rgba(30,58,138,.05) 0%, transparent 55%),
    #faf8f3 !important;
  min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1.5px solid #ede9e0 !important;
  box-shadow: 2px 0 18px rgba(26,26,46,.05) !important;
}
section[data-testid="stSidebar"] * { color: #1a1a2e !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox>div>div,
section[data-testid="stSidebar"] .stNumberInput input {
  background: #faf8f3 !important; border: 1.5px solid #ede9e0 !important;
  color: #1a1a2e !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] label {
  color: #6b6b8a !important; font-size: 11px !important;
  font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px;
}
section[data-testid="stSidebar"] hr { border-color: #ede9e0 !important; }

/* ── KPI metric cards ── */
div[data-testid="metric-container"] {
  background: #ffffff !important;
  border: 1.5px solid #ede9e0 !important;
  border-radius: 16px !important;
  padding: 18px 16px 14px !important;
  box-shadow: 0 2px 10px rgba(26,26,46,.05) !important;
  position: relative; overflow: hidden;
  transition: box-shadow .2s, transform .2s;
}
div[data-testid="metric-container"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg,#f59e0b,#f97316);
  border-radius: 16px 16px 0 0;
}
div[data-testid="metric-container"]:hover {
  box-shadow: 0 6px 22px rgba(26,26,46,.10) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="metric-container"] label {
  color: #8585a5 !important; font-family:'DM Mono',monospace !important;
  font-size: 9px !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; font-weight: 500 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
  color: #1a1a2e !important; font-family:'DM Serif Display',serif !important;
  font-size: 28px !important; font-weight: 400 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
  color: #8585a5 !important; font-family:'DM Mono',monospace !important; font-size:11px !important;
}

/* ── Train button ── */
.stButton>button {
  background: linear-gradient(135deg,#1a1a2e,#16213e) !important;
  color: #faf8f3 !important; border: none !important;
  border-radius: 10px !important; font-family:'DM Mono',monospace !important;
  font-size:13px !important; font-weight:500 !important; letter-spacing:.8px !important;
  padding: 12px 28px !important; width:100% !important;
  transition: all .2s !important;
  box-shadow: 0 4px 14px rgba(26,26,46,.18) !important;
}
.stButton>button:hover {
  background: linear-gradient(135deg,#f59e0b,#f97316) !important;
  color: #1a1a2e !important;
  box-shadow: 0 6px 22px rgba(245,158,11,.28) !important;
  transform: translateY(-1px) !important;
}

/* ── Typography ── */
h1 { font-family:'DM Serif Display',serif !important; font-size:2.7rem !important;
     font-weight:400 !important; color:#1a1a2e !important;
     letter-spacing:-1px !important; line-height:1.15 !important; }
h2 { font-family:'DM Serif Display',serif !important; font-size:1.55rem !important;
     font-weight:400 !important; color:#1a1a2e !important;
     letter-spacing:-.4px !important; margin:0 !important; }
h3 { font-family:'DM Sans',sans-serif !important; font-size:.95rem !important;
     font-weight:700 !important; color:#6b6b8a !important;
     text-transform:uppercase !important; letter-spacing:1.4px !important; }

/* ── Dividers ── */
hr { border:none !important; border-top:1.5px solid #ede9e0 !important; margin:2.2rem 0 !important; }

/* ── DataFrames ── */
.stDataFrame { border:1.5px solid #ede9e0 !important; border-radius:12px !important; overflow:hidden !important; }
.stDataFrame thead th {
  background: #f0ece3 !important; color:#1a1a2e !important;
  font-family:'DM Mono',monospace !important; font-size:10px !important; letter-spacing:1px !important;
}
.stDataFrame tbody td {
  color:#1a1a2e !important; font-family:'DM Mono',monospace !important; font-size:12px !important;
}
.stDataFrame tbody tr:hover td { background:#fdf5e4 !important; }

/* ── Sliders ── */
.stSlider>div>div>div>div { background:linear-gradient(90deg,#f59e0b,#f97316) !important; }

/* ── Progress ── */
div[data-testid="stProgressBar"]>div>div { background:linear-gradient(90deg,#f59e0b,#f97316) !important; }

/* ── Expander ── */
details { border:1.5px solid #ede9e0 !important; border-radius:10px !important;
          background:#fff !important; padding:4px 12px !important; }
details summary { color:#6b6b8a !important; font-family:'DM Mono',monospace !important;
                  font-size:11px !important; }

/* ── Signal pills ── */
.pill-long  { background:#dcfce7; color:#166534; border-radius:20px; padding:3px 14px;
              font-family:'DM Mono',monospace; font-size:12px; font-weight:600; }
.pill-short { background:#fee2e2; color:#991b1b; border-radius:20px; padding:3px 14px;
              font-family:'DM Mono',monospace; font-size:12px; font-weight:600; }
.pill-hold  { background:#f3f4f6; color:#6b7280; border-radius:20px; padding:3px 14px;
              font-family:'DM Mono',monospace; font-size:12px; font-weight:600; }

/* ── Alert boxes ── */
div[data-testid="stAlert"] {
  background:#ffffff !important; border:1.5px solid #ede9e0 !important;
  border-radius:12px !important; color:#1a1a2e !important;
  box-shadow:0 2px 8px rgba(26,26,46,.04) !important;
}

/* ── Number inputs ── */
.stNumberInput input { background:#faf8f3 !important; border-color:#ede9e0 !important; color:#1a1a2e !important; }
.stSelectbox [data-baseweb="select"]>div { background:#faf8f3 !important; border-color:#ede9e0 !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
CT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="DM Mono, monospace", color="#8585a5", size=11),
    xaxis=dict(gridcolor="#f0ece3", zerolinecolor="#ede9e0", linecolor="#ede9e0",
               tickfont=dict(color="#8585a5")),
    yaxis=dict(gridcolor="#f0ece3", zerolinecolor="#ede9e0", linecolor="#ede9e0",
               tickfont=dict(color="#8585a5")),
    margin=dict(l=12, r=12, t=44, b=12),
    hoverlabel=dict(bgcolor="#fff", bordercolor="#ede9e0",
                    font=dict(family="DM Mono, monospace", color="#1a1a2e", size=12)),
    legend=dict(bgcolor="rgba(255,255,255,.92)", bordercolor="#ede9e0", borderwidth=1,
                font=dict(family="DM Mono", size=11, color="#1a1a2e")),
)
AMBER  = "#f59e0b"
ORANGE = "#f97316"
NAVY   = "#1a1a2e"
GREEN  = "#16a34a"
RED    = "#dc2626"
SLATE  = "#94a3b8"
PURPLE = "#7c3aed"
TEAL   = "#0891b2"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def signal_badge(s):
    if s == 1:    return '<span class="pill-long">▲ LONG</span>'
    elif s == -1: return '<span class="pill-short">▼ SHORT</span>'
    return '<span class="pill-hold">● HOLD</span>'

def section_head(icon, title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:2rem 0 1rem 0;">
      <div style="width:4px;height:28px;background:linear-gradient(180deg,#f59e0b,#f97316);
                  border-radius:2px;flex-shrink:0;"></div>
      <h2>{icon}&nbsp; {title}</h2>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def price_signal_chart(df, preds):
    # !! use DatetimeIndex from df to align — preds.index is DatetimeIndex
    common_idx = df.index.intersection(preds.index)
    sub  = df.reindex(common_idx)
    prob = preds.reindex(common_idx)["pred_prob"]
    xgbp = prob  # pred_prob is already xgb raw prob
    vol  = sub["Volume"]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.56, 0.25, 0.19], vertical_spacing=0.02)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=sub.index, open=sub["Open"], high=sub["High"],
        low=sub["Low"], close=sub["Close"], name="Price",
        increasing=dict(line=dict(color=GREEN, width=1.5), fillcolor="#dcfce7"),
        decreasing=dict(line=dict(color=RED,   width=1.5), fillcolor="#fee2e2"),
    ), row=1, col=1)

    long_idx  = preds[preds["signal"] ==  1].index
    short_idx = preds[preds["signal"] == -1].index
    long_idx  = long_idx.intersection(df.index)
    short_idx = short_idx.intersection(df.index)

    if len(long_idx):
        fig.add_trace(go.Scatter(
            x=long_idx, y=df["Close"].reindex(long_idx) * 0.984,
            mode="markers", name="Long",
            marker=dict(symbol="triangle-up", size=11, color=GREEN,
                        line=dict(color="white", width=1.5)),
        ), row=1, col=1)
    if len(short_idx):
        fig.add_trace(go.Scatter(
            x=short_idx, y=df["Close"].reindex(short_idx) * 1.016,
            mode="markers", name="Short",
            marker=dict(symbol="triangle-down", size=11, color=RED,
                        line=dict(color="white", width=1.5)),
        ), row=1, col=1)

    # XGB probability ribbon (raw, well-scaled 0→1)
    fig.add_trace(go.Scatter(
        x=xgbp.index, y=xgbp.values,
        name="XGB Prob", line=dict(color=AMBER, width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
    ), row=2, col=1)
    fig.add_hline(y=np.percentile(xgbp.dropna(), 60), line_dash="dot",
                  line_color=GREEN, line_width=1, row=2, col=1)
    fig.add_hline(y=np.percentile(xgbp.dropna(), 40), line_dash="dot",
                  line_color=RED, line_width=1, row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=SLATE, line_width=1, row=2, col=1)

    # Volume
    vc = [GREEN if r >= 0 else RED for r in sub["Close"].pct_change()]
    fig.add_trace(go.Bar(x=sub.index, y=vol, name="Volume",
                         marker_color=vc, opacity=0.45, showlegend=False), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False, height=600,
        title=dict(text="Price · Signals · Confidence · Volume",
                   font=dict(family="DM Serif Display", size=15, color=NAVY), x=0.01),
        **CT,
    )
    fig.update_yaxes(title_text="Price ($)", title_font=dict(size=10, color=SLATE), row=1, col=1)
    fig.update_yaxes(title_text="Up Prob", title_font=dict(size=10, color=SLATE),
                     range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Volume", title_font=dict(size=10, color=SLATE), row=3, col=1)
    return fig


def equity_chart(results):
    eq, bheq = results["equity"], results["bh_equity"]
    dd = results["metrics"]["drawdown_series"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy",
        line=dict(color=AMBER, width=2.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bheq.index, y=bheq, name="Buy & Hold",
        line=dict(color=SLATE, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, name="Drawdown %",
        line=dict(color=RED, width=1.2),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
        showlegend=False), row=2, col=1)
    fig.update_layout(height=440,
        title=dict(text="Equity Curve vs Buy & Hold · Drawdown",
                   font=dict(family="DM Serif Display", size=15, color=NAVY), x=0.01),
        **CT)
    fig.update_yaxes(title_text="Portfolio ($)", title_font=dict(size=10, color=SLATE), row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", title_font=dict(size=10, color=SLATE), row=2, col=1)
    return fig


def returns_dist_chart(strat, bh):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=strat*100, name="Strategy", nbinsx=55,
        opacity=0.75, marker_color=AMBER, marker_line=dict(color="white", width=.5)))
    fig.add_trace(go.Histogram(x=bh*100, name="Buy & Hold", nbinsx=55,
        opacity=0.40, marker_color=SLATE, marker_line=dict(color="white", width=.5)))
    fig.update_layout(barmode="overlay", height=300,
        title=dict(text="Daily Returns Distribution (%)",
                   font=dict(family="DM Serif Display", size=14, color=NAVY), x=0.01),
        xaxis_title="Return (%)", yaxis_title="Count", **CT)
    return fig


def feature_importance_chart(fi, top_n=20):
    top  = fi.head(top_n).sort_values()
    norm = (top - top.min()) / (top.max() - top.min() + 1e-9)
    colors = [f"rgba(245,158,11,{0.30+0.70*v:.2f})" for v in norm]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in top.values], textposition="outside",
        textfont=dict(family="DM Mono", size=10, color=SLATE),
    ))
    fig.update_layout(height=max(380, top_n*22),
        title=dict(text=f"Top {top_n} Feature Importances (XGBoost)",
                   font=dict(family="DM Serif Display", size=14, color=NAVY), x=0.01),
        xaxis_title="Importance Score", **CT)
    return fig


def walkforward_chart(wf_eq):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wf_eq.index, y=wf_eq,
        name="Walk-Forward OOS Equity",
        line=dict(color=PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.07)"))
    fig.update_layout(height=360,
        title=dict(text="Walk-Forward Out-of-Sample Equity Curve",
                   font=dict(family="DM Serif Display", size=15, color=NAVY), x=0.01),
        yaxis_title="Portfolio Value ($)", **CT)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k in ["df","preds","backtest_results","feature_cols","fi","val_metrics","wf_results","horizon","_ticker"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 22px 0;">
      <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#1a1a2e;line-height:1.2;">
        StockSage</div>
      <div style="font-family:'DM Mono',monospace;font-size:9px;color:#8585a5;
                  letter-spacing:2.5px;margin-top:4px;">PREDICTION SYSTEM</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Configuration**")
    ticker  = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    period  = st.selectbox("Data Period", ["1y","2y","3y","5y","10y"], index=3)
    horizon = st.select_slider("Prediction Horizon (days)", options=[1,3,5,10,21], value=5)
    seq_len = st.select_slider("LSTM Sequence Length", options=[10,15,20,30], value=20)
    n_folds = st.slider("Walk-Forward Folds", 2, 6, 3)
    st.markdown("---")
    st.markdown("**Backtest Settings**")
    commission  = st.number_input("Commission (%)", value=0.10, step=0.05, format="%.2f") / 100
    slippage    = st.number_input("Slippage (%)",   value=0.05, step=0.01, format="%.2f") / 100
    initial_cap = st.number_input("Initial Capital ($)", value=100_000, step=10_000)
    st.markdown("---")
    run_btn = st.button("🚀  Train & Predict")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1rem;border-bottom:1.5px solid #ede9e0;">
  <div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;">
    <h1 style="margin:0;">Advanced Stock<br>
      <em style="color:#f59e0b;">Prediction System</em></h1>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8585a5;
                letter-spacing:1.5px;padding-top:6px;">
      LSTM · XGBOOST · META-STACKING<br>WALK-FORWARD · RISK-ADJUSTED
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    prog = st.progress(0, text="Fetching market data …")
    try:
        df = build_dataset(ticker, period=period, horizon=horizon)
        st.session_state.update({"df": df, "horizon": horizon, "_ticker": ticker})
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()

    feature_cols = get_feature_cols(df)
    st.session_state["feature_cols"] = feature_cols
    splits = list(walk_forward_splits(df, n_splits=n_folds))
    if not splits:
        st.error("Not enough data for the selected number of folds."); st.stop()

    train_df, val_df, test_df = splits[-1]
    prog.progress(20, text="Training LSTM + XGBoost …")

    model = EnsemblePredictor(horizon=horizon, seq_len=seq_len)
    val_metrics = model.fit(train_df, val_df, feature_cols)
    st.session_state.update({"val_metrics": val_metrics})
    prog.progress(65, text="Generating predictions …")

    preds = model.predict(test_df)
    st.session_state.update({"preds": preds, "fi": model.feature_importance()})
    prog.progress(78, text="Backtesting …")

    bt = Backtester(commission=commission, slippage=slippage, initial_capital=initial_cap)
    close_prices = df["Close"].reindex(preds.index)
    results = bt.run(preds["signal"], close_prices)
    st.session_state["backtest_results"] = results
    prog.progress(90, text="Walk-forward analysis …")

    wf_folds = []
    for tr, vl, te in splits:
        try:
            m2 = EnsemblePredictor(horizon=horizon, seq_len=seq_len)
            m2.fit(tr, vl, feature_cols)
            p2 = m2.predict(te)
            cp = df["Close"].reindex(p2.index)
            wf_folds.append((p2["signal"], cp))
        except Exception:
            pass
    if wf_folds:
        st.session_state["wf_results"] = bt.run_walkforward(wf_folds)

    prog.progress(100, text="Done ✓")
    st.success(f"✅  Model trained and backtested for **{ticker}**!")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — SINGLE SCROLLABLE PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["backtest_results"] is not None:
    results = st.session_state["backtest_results"]
    preds   = st.session_state["preds"]
    df      = st.session_state["df"]
    fi      = st.session_state["fi"]
    vm      = st.session_state["val_metrics"]
    m       = results["metrics"]
    h       = st.session_state["horizon"] or 5
    _tk     = st.session_state["_ticker"] or ""

    # ── ① KPI STRIP ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    kc = st.columns(6)
    for col, (label, val) in zip(kc, [
        ("CAGR",          f"{m['CAGR (%)']:.1f}%"),
        ("Sharpe Ratio",  f"{m['Sharpe Ratio']:.2f}"),
        ("Sortino Ratio", f"{m['Sortino Ratio']:.2f}"),
        ("Max Drawdown",  f"{m['Max Drawdown (%)']:.1f}%"),
        ("Win Rate",      f"{m['Win Rate (%)']:.1f}%"),
        ("Val Accuracy",  f"{vm['val_accuracy']*100:.1f}%"),
    ]):
        col.metric(label, val)

    # ── Latest signal banner ──────────────────────────────────────────────────
    latest   = preds.iloc[-1]
    sig_int  = int(latest["signal"])
    # Use xgb_prob if present (raw, calibrated), else display_prob
    raw_prob = float(latest["pred_prob"])  # xgb classification prob
    sig_color  = "#dcfce7" if sig_int==1 else ("#fee2e2" if sig_int==-1 else "#f3f4f6")
    sig_border = "#16a34a" if sig_int==1 else ("#dc2626" if sig_int==-1 else "#d1d5db")

    st.markdown(f"""
    <div style="margin:1.6rem 0 0;background:{sig_color};border:1.5px solid {sig_border};
                border-radius:14px;padding:18px 24px;display:flex;align-items:center;
                gap:36px;flex-wrap:wrap;">
      <div>
        <div style="font-family:'DM Mono',monospace;font-size:9px;color:#8585a5;
                    text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Latest Signal</div>
        {signal_badge(sig_int)}
      </div>
      <div>
        <div style="font-family:'DM Mono',monospace;font-size:9px;color:#8585a5;
                    text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">Up Probability (XGB)</div>
        <div style="font-family:'DM Serif Display',serif;font-size:24px;color:#1a1a2e;">
          {raw_prob*100:.1f}%
        </div>
      </div>
      <div>
        <div style="font-family:'DM Mono',monospace;font-size:9px;color:#8585a5;
                    text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">LSTM Return Est. ({h}d)</div>
        <div style="font-family:'DM Serif Display',serif;font-size:24px;color:#1a1a2e;">
          {latest['pred_return']*100:+.2f}%
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:8px;color:#b0a898;margin-top:2px;">
          regression estimate · direction from XGB
        </div>
      </div>
      <div style="margin-left:auto;font-family:'DM Mono',monospace;font-size:10px;color:#8585a5;">
        Ticker: <strong style="color:#1a1a2e;">{_tk}</strong>&nbsp;|&nbsp;
        Horizon: <strong style="color:#1a1a2e;">{h}d</strong>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── ② PRICE & SIGNALS ────────────────────────────────────────────────────
    section_head("📉", "Price & Signals")
    st.plotly_chart(price_signal_chart(df, preds), use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── ③ EQUITY CURVE + RETURNS DIST + METRICS TABLE ────────────────────────
    section_head("💹", "Equity Curve")
    eq_col, side_col = st.columns([2, 1], gap="large")

    with eq_col:
        st.plotly_chart(equity_chart(results), use_container_width=True)
        st.plotly_chart(returns_dist_chart(results["strat_ret"], results["bh_ret"]),
                        use_container_width=True)
    with side_col:
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:9px;
            color:#8585a5;text-transform:uppercase;letter-spacing:2px;
            margin-bottom:10px;">Full Backtest Metrics</div>""", unsafe_allow_html=True)
        # Exclude internal/duplicate keys
        skip = {"drawdown_series", "deflated_sharpe_p"}
        mdata = {k: v for k, v in m.items() if k not in skip}
        dsr = m.get("deflated_sharpe_p", "N/A")
        mdata["Deflated Sharpe p"] = f"{dsr} {'✓ Sig.' if isinstance(dsr,float) and dsr < 0.05 else '✗ N.S.'}" if isinstance(dsr, float) else dsr
        mdf = pd.DataFrame(list(mdata.items()), columns=["Metric","Value"])
        st.dataframe(mdf.set_index("Metric"), use_container_width=True, height=420)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── ④ FEATURE IMPORTANCE ─────────────────────────────────────────────────
    section_head("🔬", "Feature Importance")
    fi_left, fi_right = st.columns([3, 1], gap="large")
    with fi_left:
        top_n = st.slider("Top N features", 10, 40, 20, key="fi_n")
        st.plotly_chart(feature_importance_chart(fi, top_n), use_container_width=True)
    with fi_right:
        st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:9px;
            color:#8585a5;text-transform:uppercase;letter-spacing:2px;
            margin-bottom:10px;">All Features</div>""", unsafe_allow_html=True)
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Score"]
        fi_df["Score"] = fi_df["Score"].round(4)
        fi_df = fi_df.sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(fi_df, use_container_width=True, height=500)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── ⑤ PREDICTIONS TABLE  +  WALK-FORWARD ─────────────────────────────────
    pt_col, wf_col = st.columns(2, gap="large")

    with pt_col:
        section_head("📋", "Predictions Table")
        dp = preds[["signal","pred_prob","pred_return"]].copy()
        dp["pred_prob"]    = dp["pred_prob"].map("{:.1%}".format)
        dp["pred_return"]  = dp["pred_return"].map("{:+.2%}".format)
        dp["signal"]       = dp["signal"].map({1:"▲ LONG", -1:"▼ SHORT", 0:"● HOLD"})
        dp.columns         = ["Signal", "XGB Up Prob", f"LSTM {h}d Return Est."]
        if hasattr(dp.index, "strftime"):
            dp.index = dp.index.strftime("%Y-%m-%d")
        dp.index.name = "Date"
        st.dataframe(dp.sort_index(ascending=False), use_container_width=True, height=500)

    with wf_col:
        section_head("📈", "Walk-Forward Analysis")
        wf_res = st.session_state.get("wf_results")
        if wf_res and not wf_res["equity"].empty:
            st.plotly_chart(walkforward_chart(wf_res["equity"]), use_container_width=True)
            wf_m   = wf_res["metrics"]
            skip2  = {"drawdown_series"}
            skip2 = {"drawdown_series", "deflated_sharpe_p"}
            wf_data = {k: v for k, v in wf_m.items() if k not in skip2}
            dsr2 = wf_m.get("deflated_sharpe_p","N/A")
            wf_data["Deflated Sharpe p"] = f"{dsr2} {'✓ Sig.' if isinstance(dsr2,float) and dsr2 < 0.05 else '✗ N.S.'}" if isinstance(dsr2, float) else dsr2
            wf_mdf = pd.DataFrame(list(wf_data.items()), columns=["Metric","Value"])
            st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:9px;
                color:#8585a5;text-transform:uppercase;letter-spacing:2px;
                margin:12px 0 8px;">Walk-Forward Aggregate Metrics</div>""", unsafe_allow_html=True)
            st.dataframe(wf_mdf.set_index("Metric"), use_container_width=True)
        else:
            st.info("Walk-forward results not available for this run.")

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:3rem;padding:18px 0;border-top:1.5px solid #ede9e0;
                font-family:'DM Mono',monospace;font-size:9px;color:#b5b0a5;
                text-align:center;letter-spacing:2px;">
      STOCKSAGE &nbsp;·&nbsp; LSTM + XGBOOST ENSEMBLE &nbsp;·&nbsp;
      WALK-FORWARD VALIDATED &nbsp;·&nbsp; NOT FINANCIAL ADVICE
    </div>""", unsafe_allow_html=True)

else:
    # ── EMPTY / LANDING STATE ─────────────────────────────────────────────────
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:50px 0 36px;">
      <div style="font-size:60px;line-height:1;">📊</div>
      <div style="font-family:'DM Serif Display',serif;font-size:2rem;
                  color:#1a1a2e;margin:14px 0 8px;">Ready to Predict</div>
      <div style="font-family:'DM Mono',monospace;font-size:12px;color:#8585a5;letter-spacing:.4px;">
        Configure a ticker in the sidebar and click
        <strong style="color:#1a1a2e;">Train &amp; Predict</strong>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    for col, icon, title, body in [
        (c1,"🧠","LSTM + XGBoost Ensemble",
         "Bidirectional LSTM with dual heads + 500-tree XGBoost, stacked via Ridge meta-learner. Percentile-based signal thresholds ensure balanced LONG/SHORT/HOLD output."),
        (c2,"🔬","50+ Engineered Features",
         "RSI, MACD, Bollinger Bands, ATR, ADX, OBV, rolling skew/kurtosis, candlestick patterns, S&P 500 regime signals, and temporal encodings."),
        (c3,"📊","Realistic Backtesting",
         "Commission, slippage, stop-loss, one-bar execution delay, walk-forward CV, Sharpe/Sortino/Drawdown/Calmar, Deflated Sharpe significance test."),
    ]:
        col.markdown(f"""
        <div style="background:#fff;border:1.5px solid #ede9e0;border-radius:16px;
                    padding:26px 22px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;
                      background:linear-gradient(90deg,#f59e0b,#f97316);
                      border-radius:16px 16px 0 0;"></div>
          <div style="font-size:26px;margin-bottom:12px;">{icon}</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;
                      color:#1a1a2e;margin-bottom:10px;">{title}</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:13px;
                      color:#8585a5;line-height:1.65;">{body}</div>
        </div>""", unsafe_allow_html=True)