requirements.txt 
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Trading Assistant India",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 AI Trading Assistant India")
st.caption("Swing Trading | Stocks | Indices | Risk Management")
st.warning(
    "यह शैक्षणिक टूल है। संकेत गारंटीड नहीं हैं। "
    "यह लाइव ऑर्डर नहीं लगाता।"
)

SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "ITC": "ITC.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "Custom Symbol": ""
}

st.sidebar.header("Settings")

choice = st.sidebar.selectbox(
    "Stock / Index चुनें",
    list(SYMBOLS.keys())
)

if choice == "Custom Symbol":
    symbol = st.sidebar.text_input(
        "Yahoo Finance Symbol",
        value="RELIANCE.NS"
    ).strip().upper()
else:
    symbol = SYMBOLS[choice]

period = st.sidebar.selectbox(
    "Historical Data",
    ["3mo", "6mo", "1y", "2y"],
    index=1
)

capital = st.sidebar.number_input(
    "Trading Capital (₹)",
    min_value=1000,
    value=50000,
    step=1000
)

risk_percent = st.sidebar.slider(
    "Risk per Trade (%)",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.5
)

@st.cache_data(ttl=300)
def load_data(ticker, selected_period):
    data = yf.download(
        ticker,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.copy()

    for col in ["Close", "High", "Low", "Volume"]:
        if col not in data.columns:
            return pd.DataFrame()

    data["EMA20"] = data["Close"].ewm(
        span=20, adjust=False
    ).mean()

    data["EMA50"] = data["Close"].ewm(
        span=50, adjust=False
    ).mean()

    data["EMA200"] = data["Close"].ewm(
        span=200, adjust=False
    ).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(50)

    ema12 = data["Close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26, adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26
    data["SignalLine"] = data["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    previous_close = data["Close"].shift(1)

    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = tr.rolling(14).mean()

    return data.dropna(
        subset=["Close", "EMA20", "EMA50", "RSI", "ATR"]
    )

if st.sidebar.button("🔄 Refresh Data"):
    load_data.clear()

if not symbol:
    st.info("पहले कोई symbol डालें।")
    st.stop()

try:
    with st.spinner("Market data लोड हो रहा है..."):
        df = load_data(symbol, period)

    if df.empty or len(df) < 2:
        st.error(
            "डेटा नहीं मिला। Symbol जाँचें और दोबारा प्रयास करें।"
        )
        st.stop()

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])

    bullish = (
        price > ema20 > ema50
        and rsi > 50
        and latest["MACD"] > latest["SignalLine"]
    )

    bearish = (
        price < ema20 < ema50
        and rsi < 50
        and latest["MACD"] < latest["SignalLine"]
    )

    if bullish:
        signal = "BULLISH WATCH"
    elif bearish:
        signal = "BEARISH WATCH"
    else:
        signal = "WAIT / NO CLEAR SIGNAL"

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Latest Close",
        f"₹{price:,.2f}",
        f"{price - float(previous['Close']):+.2f}"
    )
    c2.metric("RSI (14)", f"{rsi:.1f}")
    c3.metric("ATR (14)", f"₹{atr:.2f}")

    st.subheader("📊 Technical Signal")

    if bullish:
        st.success(signal)
    elif bearish:
        st.error(signal)
    else:
        st.info(signal)

    st.caption(
        f"Latest available daily candle: "
        f"{df.index[-1].strftime('%d-%m-%Y')}"
    )

    st.subheader("📈 Price & Moving Averages")

    chart = df[["Close", "EMA20", "EMA50", "EMA200"]].tail(120)
    st.line_chart(chart)

    st.subheader("🧮 Illustrative Risk Calculator")

    risk_amount = capital * risk_percent / 100
    stop_distance = 2 * atr
    stop_loss = price - stop_distance
    target = price + (2 * stop_distance)

    quantity_by_risk = int(risk_amount / stop_distance)
    quantity_by_capital = int(capital / price)
    quantity = max(
        0, min(quantity_by_risk, quantity_by_capital)
    )

    st.write(f"Risk budget: ₹{risk_amount:,.2f}")
    st.write(f"Illustrative stop-loss: ₹{stop_loss:,.2f}")
    st.write(f"Illustrative target: ₹{target:,.2f}")
    st.write(f"Illustrative quantity: {quantity} shares")
    st.caption(
        "ये केवल गणना के उदाहरण हैं, खरीदने/बेचने की सलाह नहीं। "
        "Gap, slippage, charges और वास्तविक execution से नुकसान "
        "निर्धारित जोखिम से अधिक हो सकता है।"
    )

    st.subheader("📋 Recent Market Data")
    st.dataframe(
        df[["Close", "EMA20", "EMA50", "RSI", "MACD", "ATR"]]
        .tail(10)
        .sort_index(ascending=False),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Historical Data CSV",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{symbol.replace('.', '_')}_data.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(
        "डेटा लोड नहीं हो पाया। इंटरनेट या symbol जाँचें, "
        "फिर Refresh Data दबाएँ।"
    )
    st.caption(f"तकनीकी विवरण: {e}")

