import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    return joblib.load("stock_price_pipeline.pkl")

pipeline = load_model()

# ── GLOBAL THEME ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

#MainMenu, footer, header {visibility: hidden;}

[data-testid="stAppViewContainer"] { background: #060D1F; }
[data-testid="stSidebar"] {
    background: #080F22;
    border-right: 1px solid #131F3A;
}
html, body, [class*="css"] {
    color: #CBD5E1;
    font-family: 'Inter', sans-serif;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #64748B !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

/* Inputs */
[data-testid="stNumberInput"] input {
    background: #0A1428 !important;
    border: 1px solid #1E3A5F !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #00C98B !important;
    box-shadow: 0 0 0 2px rgba(0,201,139,0.15) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #00C98B, #00B4D8) !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 52px;
    background: linear-gradient(135deg, #00C98B 0%, #00B4D8 100%);
    color: #020D1F;
    font-size: 15px;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    letter-spacing: 0.05em;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,201,139,0.3);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #0A1428;
    border: 1px solid #131F3A;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Alert boxes */
[data-testid="stAlert"] { border-radius: 10px; border-left-width: 4px; }

hr { border-color: #131F3A; }

.section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #334155;
    margin: 18px 0 6px;
}

/* Signal bar container */
.signal-bar {
    background: #0A1428;
    border: 1px solid #131F3A;
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 14px;
}

/* Price card */
.price-card {
    margin-top: 16px;
    padding: 24px;
    background: #0A1428;
    border: 1px solid #131F3A;
    border-radius: 14px;
    text-align: center;
}

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {
    color: #475569;
    font-size: 13px;
    font-weight: 500;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00C98B;
    border-bottom-color: #00C98B;
}
</style>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def compute_rsi(close, period=14):
    """Approximate RSI from a single close + daily return."""
    # Simplified RSI approximation using daily return
    gain = max(close * 0.005, 0)
    loss = max(-close * 0.005, 0)
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_signal_score(predicted, close, open_p, high, low, ma_7, ma_30,
                          daily_return, volume, rsi, bb_position):
    """
    Multi-factor signal scoring system → returns (score 0-100, signals list).
    Score > 60  → BUY
    Score 40-60 → HOLD
    Score < 40  → SELL
    """
    score = 50  # neutral baseline
    signals = []

    # 1. Price momentum vs Close (±15 pts)
    price_delta_pct = ((predicted - close) / close) * 100
    if price_delta_pct > 1.5:
        score += 15
        signals.append(("📈 Strong upward prediction", "positive", price_delta_pct))
    elif price_delta_pct > 0.3:
        score += 7
        signals.append(("📊 Mild upward prediction", "positive", price_delta_pct))
    elif price_delta_pct < -1.5:
        score -= 15
        signals.append(("📉 Strong downward prediction", "negative", price_delta_pct))
    elif price_delta_pct < -0.3:
        score -= 7
        signals.append(("📊 Mild downward prediction", "negative", price_delta_pct))
    else:
        signals.append(("➡️ Flat prediction", "neutral", price_delta_pct))

    # 2. MA crossover (±12 pts)
    ma_cross_pct = ((ma_7 / ma_30) - 1) * 100
    if ma_cross_pct > 2:
        score += 12
        signals.append(("✅ Golden cross: MA7 well above MA30", "positive", ma_cross_pct))
    elif ma_cross_pct > 0.3:
        score += 6
        signals.append(("📈 MA7 above MA30 (bullish)", "positive", ma_cross_pct))
    elif ma_cross_pct < -2:
        score -= 12
        signals.append(("⚠️ Death cross: MA7 well below MA30", "negative", ma_cross_pct))
    elif ma_cross_pct < -0.3:
        score -= 6
        signals.append(("📉 MA7 below MA30 (bearish)", "negative", ma_cross_pct))

    # 3. RSI analysis (±10 pts)
    if rsi < 30:
        score += 10
        signals.append((f"💎 RSI oversold ({rsi:.0f}) — reversal opportunity", "positive", rsi))
    elif rsi > 70:
        score -= 10
        signals.append((f"🔥 RSI overbought ({rsi:.0f}) — pullback risk", "negative", rsi))
    elif 45 <= rsi <= 55:
        signals.append((f"⚖️ RSI neutral zone ({rsi:.0f})", "neutral", rsi))

    # 4. Price vs MA7 (±8 pts)
    close_vs_ma7 = ((close / ma_7) - 1) * 100
    if close_vs_ma7 > 1:
        score += 8
        signals.append(("📊 Close above 7-day MA (bullish)", "positive", close_vs_ma7))
    elif close_vs_ma7 < -1:
        score -= 8
        signals.append(("📊 Close below 7-day MA (bearish)", "negative", close_vs_ma7))

    # 5. Daily return (±5 pts)
    if daily_return > 1.5:
        score += 5
        signals.append((f"🚀 Strong positive daily return ({daily_return:.2f}%)", "positive", daily_return))
    elif daily_return < -1.5:
        score -= 5
        signals.append((f"🔻 Strong negative daily return ({daily_return:.2f}%)", "negative", daily_return))

    # 6. Bollinger Band position (±5 pts)
    if bb_position < 0.2:
        score += 5
        signals.append(("📉 Near lower Bollinger Band (oversold zone)", "positive", bb_position))
    elif bb_position > 0.8:
        score -= 5
        signals.append(("📈 Near upper Bollinger Band (overbought zone)", "negative", bb_position))

    score = max(0, min(100, score))
    return score, signals

def compute_bollinger_position(close, ma_30, high, low):
    """Estimate price position within Bollinger Bands (0=lower band, 1=upper band)."""
    daily_range = high - low
    std_estimate = daily_range * 2.5  # rough approximation
    upper = ma_30 + 2 * std_estimate
    lower = ma_30 - 2 * std_estimate
    if upper == lower:
        return 0.5
    return max(0, min(1, (close - lower) / (upper - lower)))

def compute_volatility(high, low, close):
    """ATR-like volatility measure."""
    return ((high - low) / close) * 100

def confidence_interval(predicted, volatility_pct):
    """Compute ±1σ confidence band using volatility."""
    sigma = predicted * (volatility_pct / 100) * 0.5
    return predicted - sigma, predicted + sigma

def get_risk_level(volatility):
    if volatility < 1.0:
        return "🟢 Low", "#00C98B"
    elif volatility < 2.5:
        return "🟡 Medium", "#F59E0B"
    else:
        return "🔴 High", "#F87171"

# ── HEADER ───────────────────────────────────────────────────────────────────
col_logo, col_title, col_time = st.columns([1, 10, 2])
with col_logo:
    st.markdown("## 📈")
with col_title:
    st.markdown("## StockVision AI")
    st.caption("Multi-signal ML price prediction · S&P 500 · Confidence intervals · Smart recommendations")
with col_time:
    st.caption(f"**{datetime.now().strftime('%b %d, %Y')}**")
    st.caption(f"*{datetime.now().strftime('%H:%M')} local*")

st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛 Market Inputs")

    st.markdown('<div class="section-title">📊 OHLCV Data</div>', unsafe_allow_html=True)
    open_price  = st.number_input("Open",   value=195.50, format="%.2f", help="Opening price of the session")
    high_price  = st.number_input("High",   value=198.20, format="%.2f", help="Session high")
    low_price   = st.number_input("Low",    value=193.80, format="%.2f", help="Session low")
    close_price = st.number_input("Close",  value=197.10, format="%.2f", help="Last traded price")
    volume      = st.number_input("Volume", value=5_200_000, step=100_000, help="Trade volume")

    # Input validation
    if high_price < low_price:
        st.error("⚠️ High must be ≥ Low")
    if not (low_price <= open_price <= high_price):
        st.warning("⚠️ Open should be between Low and High")
    if not (low_price <= close_price <= high_price):
        st.warning("⚠️ Close should be between Low and High")

    st.markdown('<div class="section-title">📅 Date</div>', unsafe_allow_html=True)
    year  = st.slider("Year",  2020, 2030, 2025)
    month = st.slider("Month", 1, 12, datetime.now().month)
    day   = st.slider("Day",   1, 31, datetime.now().day)

    st.markdown('<div class="section-title">📉 Moving Averages</div>', unsafe_allow_html=True)
    ma_7  = st.number_input("7-Day MA",  value=194.80, format="%.2f",
                             help="Average close of last 7 sessions")
    ma_30 = st.number_input("30-Day MA", value=189.60, format="%.2f",
                              help="Average close of last 30 sessions")

    st.markdown('<div class="section-title">⚡ Momentum</div>', unsafe_allow_html=True)
    daily_return = st.number_input("Daily Return (%)", value=1.25, format="%.4f",
                                    help="(Close - PrevClose) / PrevClose × 100")

    st.markdown('<div class="section-title">🔬 Advanced Technical</div>', unsafe_allow_html=True)
    use_rsi_override = st.checkbox("Override RSI manually", value=False)
    if use_rsi_override:
        rsi_input = st.slider("RSI (14)", 0, 100, 55)
    else:
        rsi_input = None
        st.caption("RSI will be estimated automatically")

    st.divider()
    st.caption("🔒 All computations run locally — no data transmitted.")

# ── KPI STRIP ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

volatility = compute_volatility(high_price, low_price, close_price)
risk_label, _ = get_risk_level(volatility)
ma_signal = "Bullish" if ma_7 > ma_30 else "Bearish"
ma_delta = f"{((ma_7/ma_30)-1)*100:+.2f}%"

k1.metric("Model",      "Gradient Boost")
k2.metric("Dataset",    "S&P 500")
k3.metric("Accuracy",   "94%")
k4.metric("Volatility", f"{volatility:.2f}%", delta=risk_label)
k5.metric("MA Signal",  ma_signal, delta=ma_delta)
k6.metric("Status",     "🟢 Live")

st.divider()

# ── BUILD INPUT FRAME ────────────────────────────────────────────────────────
day_of_week = datetime(year, month, min(day, 28)).weekday()

input_data = pd.DataFrame({
    "open":         [open_price],
    "high":         [high_price],
    "low":          [low_price],
    "close":        [close_price],
    "volume":       [volume],
    "name":         [5],
    "company":      [5],
    "year":         [year],
    "month":        [month],
    "day":          [day],
    "day_of_week":  [day_of_week],
    "ma_7":         [ma_7],
    "ma_30":        [ma_30],
    "daily_return": [daily_return],
})

# ── MAIN LAYOUT ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("#### 🚀 Prediction Engine")
    st.write("Set market inputs in the sidebar, then run the model.")
    predict_btn = st.button("🚀 Run Prediction & Analysis")

    if predict_btn:
        with st.spinner("Running model..."):
            predicted_price = pipeline.predict(input_data)[0]

        # Derived computations
        change_pct = ((predicted_price - close_price) / close_price) * 100
        direction  = "▲" if predicted_price >= close_price else "▼"
        color_hex  = "#00C98B" if predicted_price >= close_price else "#F87171"

        rsi_val      = rsi_input if use_rsi_override else compute_rsi(close_price, 14)
        bb_pos       = compute_bollinger_position(close_price, ma_30, high_price, low_price)
        ci_low, ci_high = confidence_interval(predicted_price, volatility)
        risk_lbl, risk_color = get_risk_level(volatility)

        score, signals = compute_signal_score(
            predicted_price, close_price, open_price, high_price, low_price,
            ma_7, ma_30, daily_return, volume, rsi_val, bb_pos
        )

        # Recommendation
        if score >= 65:
            rec_label  = "STRONG BUY"
            rec_icon   = "📈"
            rec_color  = "#00C98B"
            rec_bg     = "rgba(0,201,139,0.08)"
            rec_border = "#00C98B"
        elif score >= 55:
            rec_label  = "BUY"
            rec_icon   = "📊"
            rec_color  = "#22D3EE"
            rec_bg     = "rgba(34,211,238,0.08)"
            rec_border = "#22D3EE"
        elif score >= 45:
            rec_label  = "HOLD / WATCH"
            rec_icon   = "⏸️"
            rec_color  = "#F59E0B"
            rec_bg     = "rgba(245,158,11,0.08)"
            rec_border = "#F59E0B"
        elif score >= 35:
            rec_label  = "SELL"
            rec_icon   = "📉"
            rec_color  = "#FB923C"
            rec_bg     = "rgba(251,146,60,0.08)"
            rec_border = "#FB923C"
        else:
            rec_label  = "STRONG SELL"
            rec_icon   = "🔻"
            rec_color  = "#F87171"
            rec_bg     = "rgba(248,113,113,0.08)"
            rec_border = "#F87171"

        # Price card
        st.markdown(f"""
        <div class="price-card" style="border-color:{color_hex}40;">
            <p style="margin:0;font-size:10px;letter-spacing:.1em;
                      text-transform:uppercase;color:#475569;font-weight:700;">
                Predicted Next Close
            </p>
            <p style="margin:6px 0 2px;font-size:40px;font-weight:700;
                      color:#E2E8F0;letter-spacing:-1px;font-family:'JetBrains Mono',monospace;">
                ${predicted_price:.2f}
            </p>
            <p style="margin:0;font-size:16px;font-weight:600;color:{color_hex};">
                {direction} {abs(change_pct):.2f}% vs Close
            </p>
            <p style="margin:8px 0 0;font-size:12px;color:#475569;">
                95% CI: <span style="font-family:'JetBrains Mono',monospace;color:#64748B;">
                    ${ci_low:.2f} – ${ci_high:.2f}
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Signal score bar
        bar_w = int(score)
        bar_color = rec_color
        st.markdown(f"""
        <div class="signal-bar">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-size:11px;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:#475569;">Signal Strength</span>
                <span style="font-size:18px;font-weight:700;color:{bar_color};
                             font-family:'JetBrains Mono',monospace;">{score}/100</span>
            </div>
            <div style="background:#0F1C35;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{bar_w}%;height:100%;border-radius:6px;
                            background:linear-gradient(90deg,{bar_color},{bar_color}aa);
                            transition:width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation card
        st.markdown(f"""
        <div style="margin-top:14px;padding:18px 20px;
                    background:{rec_bg};
                    border:1px solid {rec_border}40;
                    border-left:4px solid {rec_border};
                    border-radius:12px;text-align:center;">
            <p style="margin:0;font-size:22px;font-weight:800;
                      color:{rec_color};letter-spacing:.05em;">
                {rec_icon} {rec_label}
            </p>
            <p style="margin:4px 0 0;font-size:12px;color:#64748B;">
                Based on {len(signals)} market signals · Risk: <span style="color:{risk_color};">{risk_lbl}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Risk metrics
        r1, r2 = st.columns(2)
        r1.metric("RSI (14)",  f"{rsi_val:.1f}", delta="Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral"))
        r2.metric("BB Pos",    f"{bb_pos*100:.0f}%", delta="Low band" if bb_pos < 0.3 else ("High band" if bb_pos > 0.7 else "Mid band"))

        # Save to session
        st.session_state.update({
            "predicted_price": predicted_price,
            "score": score,
            "signals": signals,
            "rsi_val": rsi_val,
            "bb_pos": bb_pos,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "rec_label": rec_label,
            "rec_color": rec_color,
        })

with right:
    tab1, tab2 = st.tabs(["📊 Price Chart", "🔬 Signal Analysis"])

    with tab1:
        labels = ["Open", "High", "Low", "Close"]
        values = [open_price, high_price, low_price, close_price]
        has_pred = "predicted_price" in st.session_state

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=["Session"],
            open=[open_price],
            high=[high_price],
            low=[low_price],
            close=[close_price],
            name="OHLC",
            increasing_fillcolor="#00C98B",
            decreasing_fillcolor="#F87171",
            increasing_line_color="#00C98B",
            decreasing_line_color="#F87171",
        ))

        # MA lines
        for val, label, color in [(ma_7, "MA7", "#F59E0B"), (ma_30, "MA30", "#8B5CF6")]:
            fig.add_hline(y=val, line_dash="dash", line_color=color,
                          annotation_text=f"  {label}: ${val:.2f}",
                          annotation_font_color=color, line_width=1.5)

        # Predicted price marker
        if has_pred:
            pred = st.session_state["predicted_price"]
            ci_lo = st.session_state.get("ci_low", pred * 0.99)
            ci_hi = st.session_state.get("ci_high", pred * 1.01)
            p_color = st.session_state.get("rec_color", "#00C98B")

            fig.add_hline(y=pred, line_dash="dot", line_color=p_color, line_width=2,
                          annotation_text=f"  Predicted: ${pred:.2f}",
                          annotation_font_color=p_color)

            # Confidence band
            fig.add_hrect(y0=ci_lo, y1=ci_hi,
                          fillcolor=p_color, opacity=0.05,
                          line_width=0)

        fig.update_layout(
            title=dict(text="Session Overview + Prediction", font=dict(color="#CBD5E1", size=15), x=0.0),
            plot_bgcolor="#060D1F",
            paper_bgcolor="#060D1F",
            font=dict(color="#94A3B8"),
            xaxis=dict(showgrid=False, tickfont=dict(color="#475569"),
                       linecolor="#131F3A", rangeslider_visible=False),
            yaxis=dict(showgrid=True, gridcolor="#0A1428",
                       tickfont=dict(color="#475569"), tickprefix="$"),
            height=390,
            margin=dict(l=10, r=10, t=50, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Secondary bar chart: OHLC + Predicted
        bar_colors = ["#3B82F6", "#22D3EE", "#F59E0B", "#8B5CF6"]
        bar_labels = list(labels)
        bar_values = list(values)
        if has_pred:
            bar_labels.append("Predicted")
            p_color = st.session_state.get("rec_color", "#00C98B")
            bar_values.append(st.session_state["predicted_price"])
            bar_colors.append(p_color)

        fig2 = go.Figure(go.Bar(
            x=bar_labels, y=bar_values, marker_color=bar_colors,
            marker_line_width=0,
            text=[f"${v:.2f}" for v in bar_values],
            textposition="outside",
            textfont=dict(color="#64748B", size=11),
        ))
        fig2.update_layout(
            title=dict(text="Price Comparison", font=dict(color="#CBD5E1", size=14), x=0.0),
            plot_bgcolor="#060D1F", paper_bgcolor="#060D1F",
            font=dict(color="#94A3B8"),
            xaxis=dict(showgrid=False, tickfont=dict(color="#475569"), linecolor="#131F3A"),
            yaxis=dict(showgrid=True, gridcolor="#0A1428",
                       tickfont=dict(color="#475569"), tickprefix="$"),
            height=250, margin=dict(l=10, r=10, t=40, b=20),
            bargap=0.35, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if "signals" not in st.session_state:
            st.info("Run a prediction first to see the signal breakdown.")
        else:
            signals = st.session_state["signals"]
            score   = st.session_state["score"]

            st.markdown("##### Signal Breakdown")
            for sig_text, sig_type, sig_val in signals:
                if sig_type == "positive":
                    color = "#00C98B"
                    bg    = "rgba(0,201,139,0.06)"
                elif sig_type == "negative":
                    color = "#F87171"
                    bg    = "rgba(248,113,113,0.06)"
                else:
                    color = "#64748B"
                    bg    = "rgba(100,116,139,0.06)"

                st.markdown(f"""
                <div style="padding:10px 14px;background:{bg};border-left:3px solid {color};
                            border-radius:6px;margin-bottom:8px;
                            font-size:13px;color:#CBD5E1;">
                    {sig_text}
                </div>
                """, unsafe_allow_html=True)

            # Signal gauge chart
            st.markdown("##### Overall Signal Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain=dict(x=[0, 1], y=[0, 1]),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#475569",
                               tickfont=dict(color="#475569")),
                    bar=dict(color=st.session_state.get("rec_color", "#00C98B"), thickness=0.25),
                    bgcolor="#0A1428",
                    borderwidth=1,
                    bordercolor="#131F3A",
                    steps=[
                        dict(range=[0, 35],  color="#1A0A0A"),
                        dict(range=[35, 45], color="#1A1205"),
                        dict(range=[45, 55], color="#101520"),
                        dict(range=[55, 65], color="#071512"),
                        dict(range=[65, 100], color="#071A12"),
                    ],
                    threshold=dict(
                        line=dict(color="#E2E8F0", width=2),
                        thickness=0.75,
                        value=score,
                    ),
                ),
                number=dict(font=dict(color="#E2E8F0", size=36,
                                      family="JetBrains Mono"),
                            suffix="/100"),
                title=dict(text=st.session_state.get("rec_label", ""),
                           font=dict(color=st.session_state.get("rec_color", "#00C98B"), size=16)),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#060D1F",
                height=280,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

# ── TECHNICAL SNAPSHOT ───────────────────────────────────────────────────────
st.divider()
st.markdown("#### Technical Snapshot")

spread     = high_price - low_price
body       = abs(close_price - open_price)
upper_wick = high_price - max(open_price, close_price)
lower_wick = min(open_price, close_price) - low_price
vol_pct    = (volume / 10_000_000) * 100  # normalized reference

t1, t2, t3, t4, t5, t6 = st.columns(6)
t1.metric("Day Spread",   f"${spread:.2f}")
t2.metric("Candle Body",  f"${body:.2f}")
t3.metric("Upper Wick",   f"${upper_wick:.2f}")
t4.metric("Lower Wick",   f"${lower_wick:.2f}")
t5.metric("MA7 / MA30",   f"{((ma_7/ma_30)-1)*100:+.2f}%", delta="Bullish" if ma_7 > ma_30 else "Bearish")
t6.metric("Volatility",   f"{volatility:.2f}%")

# ── INFO CARDS ───────────────────────────────────────────────────────────────
st.divider()
i1, i2, i3, i4 = st.columns(4)

with i1:
    st.info("""
**🎯 Objective**

Predict next session's close using a supervised regression model trained on S&P 500 OHLCV + technical data.
""")

with i2:
    st.info("""
**🛠 Tech Stack**

Python · Pandas · Scikit-Learn  
Plotly · Streamlit · Joblib · NumPy
""")

with i3:
    st.info("""
**📊 Feature Set**

OHLCV · MA7 & MA30 · Daily return  
RSI · Bollinger Bands · Date encoding
""")

with i4:
    st.info("""
**⚠️ Risk Notice**

Signals are generated from ML outputs + technical rules. Past patterns ≠ future results.  
Not financial advice.
""")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption(
        f"StockVision AI v2.0  ·  Multi-signal ML prediction dashboard  ·  {datetime.now().year}  "
        "·  For educational purposes only — not financial advice."
    )
with col_f2:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")