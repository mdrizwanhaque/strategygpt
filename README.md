# 📈 StrategyGPT: test trading ideas in plain English

**Arthaton 2026 · AI for Finance**

Most students have trading ideas ("buy Nifty when RSI drops below 30") but can't code a backtest to check them. StrategyGPT lets you describe a strategy in plain English. AI turns it into precise rules, the app backtests them on real NSE history, and the AI explains the results in simple words.

**Live app:** https://strategygpt.streamlit.app
**Demo video:** ADD_YOUTUBE_LINK_HERE

## What it does
- Type a strategy in plain English, or pick an example
- AI converts it into structured entry and exit rules (SMA, EMA, RSI, crossovers)
- Every AI-generated rule is validated before it runs, so the model can't run unsafe or made-up code
- Backtests on historical NSE data (Nifty 50, Reliance, TCS, any `.NS` ticker) with trading costs
- Compares against buy-and-hold: total return, CAGR, max drawdown, Sharpe ratio, number of trades, time in market
- Chart showing the growth of ₹1,00,000
- AI explains the results and gives an honest risk warning

Historical data only. No real money, no investment advice.

## How to run
```bash
git clone https://github.com/mdrizwanhaque/strategygpt.git
cd strategygpt
pip install -r requirements.txt
mkdir .streamlit
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml
streamlit run app.py
```
Get a free Gemini API key at https://aistudio.google.com/apikey. The three example strategies also work without a key.

## AI models and APIs used
- **Google Gemini 2.5 Flash** (via the `google-genai` SDK): turns English into JSON strategy rules and explains backtest results
- **Yahoo Finance** (via `yfinance`): historical NSE price data

## How it works
1. Gemini returns the strategy as JSON using only allowed features (`close`, `sma_N`, `ema_N`, `rsi_N`) and operators (`>`, `<`, `crosses_above`, `crosses_below`)
2. The app validates every rule and rejects anything unknown
3. Signals are shifted by one day to avoid look-ahead bias
4. Trading costs (basis points per trade) are deducted on every buy and sell

## Tech stack
Python, Streamlit, pandas, NumPy, Plotly, yfinance, Google Gemini

## Credits
Open-source libraries: Streamlit, pandas, NumPy, Plotly, yfinance, google-genai. Built by Md Rizwan Haque for Arthaton 2026.
