import ta

def add_indicators(df):

    # Ensure Close column is 1D
    close = df["Close"].squeeze()

    # Moving Average
    df["MA20"] = close.rolling(window=20).mean()

    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
    df["RSI"] = rsi_indicator.rsi()

    return df