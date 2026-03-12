# AI Stock Predictor 📈

An interactive **Stock Market Analytics Dashboard** built with Python and Streamlit that provides real-time stock insights, technical indicators, and AI-based price prediction.

The application fetches live stock market data, visualizes price trends, and applies machine learning models to estimate future stock prices.

---

## Features

- 📊 **Live Market Prices**
  - Displays real-time prices for major stocks like Apple, Tesla, Microsoft, Google, and Amazon.

- 📈 **Stock Price Trend Visualization**
  - Interactive chart showing historical stock price movement.

- 📉 **Technical Indicators**
  - Moving Average (MA20)
  - Relative Strength Index (RSI)

- 🤖 **AI Price Prediction**
  - Uses a machine learning model to estimate future stock prices.

- 📊 **Trading Volume Visualization**
  - Displays market activity with volume bar charts.

- 🌙 **Modern Dark Dashboard**
  - Clean financial dashboard interface built using Streamlit.

---

## Tech Stack

- Python
- Streamlit
- Yahoo Finance API (yfinance)
- Pandas
- NumPy
- Plotly
- Scikit-learn
- TensorFlow / LSTM

---

## Project Structure

```
AI-Stock-Predictor
│
├── app.py              # Main Streamlit dashboard
├── indicators.py       # Technical indicators (RSI, Moving Average)
├── lstm_model.py       # AI prediction model
├── requirements.txt    # Dependencies
└── README.md
```

## Installation

Follow these steps to run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/inceptioncodes/AI-Stock-Predictor.git
```

### 2. Navigate to the project directory
```bash
cd AI-Stock-Predictor
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Mac / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Run the application
```bash
streamlit run app.py
```

### 7. Open the dashboard

After running the command above, open the URL shown in the terminal (usually):

```
http://localhost:8501
```


---
Author

Dev
GitHub: https://github.com/inceptioncodes

License

This project is open-source and available under the MIT License.

