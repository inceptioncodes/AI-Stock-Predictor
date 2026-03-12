import streamlit as st
import yfinance as yf
import datetime
from indicators import add_indicators
try:
    from lstm_model import get_data, train_lstm
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False
# Page config
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}

[data-testid="stHeader"] {
    background-color: #0e1117;
}

[data-testid="stToolbar"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

# Dark theme
st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color:grey; text-align:center;'>AI Stock Predictor</h1>", unsafe_allow_html=True)

# Live market ticker
st.subheader("Live Market Prices")
tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

cols = st.columns(len(tickers))

for i, t in enumerate(tickers):

    data = yf.download(t, period="1d", interval="1m")

    if not data.empty:

        data.columns = data.columns.get_level_values(0)

        price = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[0])

        change = price - prev

        cols[i].metric(
            label=t,
            value=f"${round(price,2)}",
            delta=round(change,2)
        )

    else:

        cols[i].metric(
            label=t,
            value="No Data",
            delta="0"
        )
# Market status
import pytz

ny = pytz.timezone("America/New_York")
now = datetime.datetime.now(ny)

market_open = now.replace(hour=9, minute=30, second=0)
market_close = now.replace(hour=16, minute=0, second=0)

if market_open <= now <= market_close:
    status = "🟢 Market Status: OPEN"
else:
    status = "🔴 Market Status: CLOSED"
st.markdown(
    f"""
    <div style="
    background-color:#2f3136;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:18px;
    text-align:center;
    ">
    {status}
    </div>
    """,
    unsafe_allow_html=True
)

# Stock selector
ticker = st.selectbox(
    "Select Stock",
    ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
)

# Download data
with st.spinner("Loading market data..."):
    df = yf.download(ticker, period="1y")

# Fix MultiIndex columns
df.columns = df.columns.get_level_values(0)

# Add indicators
df = add_indicators(df)

df = df.dropna()

# Clean float values
current_price = float(df["Close"].iloc[-1])
previous_price = float(df["Close"].iloc[-2])

price_change = current_price - previous_price

# Metrics
col1, col2 = st.columns(2)

col1.metric(
    "Current Price",
    f"${round(current_price,2)}",
    delta=f"{round(price_change,2)}"
)

col2.metric(
    "Moving Average",
    f"${round(float(df['MA20'].iloc[-1]),2)}"
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.subheader("Stock Price Trend")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3]
)

# Price line
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Stock Price",
        line=dict(color="#00ffcc", width=3)
    ),
    row=1,
    col=1
)

# Moving Average
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA20"],
        mode="lines",
        name="MA20",
        line=dict(color="orange", width=2)
    ),
    row=1,
    col=1
)

# Volume bars
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color="gray"
    ),
    row=2,
    col=1
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    showlegend=True
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": False}
)
# RSI indicator
st.subheader("RSI Indicator")

st.line_chart(df["RSI"])
st.subheader("AI Stock Prediction (LSTM)")
if AI_AVAILABLE:

    X, y, scaler, data = get_data(ticker)
    model = train_lstm(X, y)

    last_60 = X[-1].reshape(1, X.shape[1], 1)

    prediction = model.predict(last_60)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    st.metric("AI Predicted Next Price", f"${predicted_price:.2f}")

else:
    st.info("AI prediction module not available in deployed environment.")

# Dataset preview
st.subheader("Dataset Preview")

st.dataframe(df.tail())