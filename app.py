"""StrategyGPT: describe a trading strategy in plain English, backtest it on NSE history.
Historical data only. Not investment advice."""
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="StrategyGPT", page_icon="📈", layout="wide")

# Tried in order; the first one Google accepts is used
MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
API = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM = """You convert a long-only trading strategy written in plain English into JSON.
Allowed features: close, sma_N, ema_N, rsi_N (N is a whole number from 2 to 200).
Allowed ops: ">", "<", "crosses_above", "crosses_below".
"right" may be a feature or a number.
Return ONLY this JSON:
{"name": str,
 "entry": [{"left": feature, "op": op, "right": feature_or_number}],
 "exit":  [{"left": feature, "op": op, "right": feature_or_number}],
 "explanation": "one plain-English sentence"}
Buy when ALL entry conditions are true. Sell when ANY exit condition is true.
If the user gives no exit rule, add a sensible opposite one."""

EXAMPLES = {
    "Golden cross": "Buy when the 50-day average crosses above the 200-day average, sell when it crosses back below.",
    "RSI dip buyer": "Buy when 14-day RSI drops below 30 and sell when it goes above 60.",
    "Trend + momentum": "Buy when price is above the 100 day EMA and RSI 14 is above 55. Exit when price falls below the 100 day EMA.",
}

# Offline fallbacks so the demo works even without an API key
PRESETS = {
    EXAMPLES["Golden cross"]: {
        "name": "Golden cross",
        "entry": [{"left": "sma_50", "op": "crosses_above", "right": "sma_200"}],
        "exit": [{"left": "sma_50", "op": "crosses_below", "right": "sma_200"}],
        "explanation": "Hold while the 50-day average is above the 200-day average.",
    },
    EXAMPLES["RSI dip buyer"]: {
        "name": "RSI dip buyer",
        "entry": [{"left": "rsi_14", "op": "<", "right": 30}],
        "exit": [{"left": "rsi_14", "op": ">", "right": 60}],
        "explanation": "Buy oversold dips and sell once momentum recovers.",
    },
    EXAMPLES["Trend + momentum"]: {
        "name": "Trend + momentum",
        "entry": [{"left": "close", "op": ">", "right": "ema_100"},
                  {"left": "rsi_14", "op": ">", "right": 55}],
        "exit": [{"left": "close", "op": "<", "right": "ema_100"}],
        "explanation": "Ride strong uptrends and step aside when the trend breaks.",
    },
}

FEAT = re.compile(r"^(close|(sma|ema|rsi)_(\d{1,3}))$")
OPS = {">", "<", "crosses_above", "crosses_below"}


# ---------- AI ----------
def has_key() -> bool:
    try:
        return bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        return False


def _key() -> str:
    return str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("\u201c\u201d").strip()


def _models(key: str) -> list:
    """Ask Google which models this key can use, flash models first."""
    import requests
    found = []
    try:
        r = requests.get(f"{API}/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=8)
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                found.append(m["name"].split("/", 1)[1])
    except Exception:
        pass
    flash = [m for m in found if "flash" in m and "image" not in m and "tts" not in m and "live" not in m]
    return list(dict.fromkeys(MODELS + flash + found))


def ask_llm(prompt: str, system: str, as_json: bool) -> str:
    import requests
    key = _key()
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    gen = {"responseMimeType": "application/json"} if as_json else {}
    errors = []
    for model in _models(key)[:4]:
        cfg = dict(gen)
        if "2.5" in model or "latest" in model:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}  # fast answers, no long thinking
        body["generationConfig"] = cfg
        try:
            r = requests.post(f"{API}/models/{model}:generateContent",
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=20)
        except requests.RequestException as e:
            errors.append(f"{model}: {type(e).__name__}")
            continue
        if r.status_code == 400 and "thinking" in r.text.lower():
            body["generationConfig"] = gen
            r = requests.post(f"{API}/models/{model}:generateContent",
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=20)
        if r.ok:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        msg = r.json().get("error", {}).get("message", r.text)[:160] if r.text else str(r.status_code)
        errors.append(f"{model}: {r.status_code} {msg}")
        if r.status_code in (400, 401, 403) and "not found" not in msg.lower():
            break  # key problem, not a model problem
    raise RuntimeError(f"Gemini refused (key starts {key[:4]}, length {len(key)}). " + " | ".join(errors))



# ---------- offline parser for common phrasings ----------
_NUM = r"(\d{1,3}(?:\.\d+)?)"


def _feat(txt: str):
    t = txt.lower()
    m = re.search(r"rsi\s*\(?\s*(\d{1,3})|(\d{1,3})[\s-]*(?:day|period)?\s*rsi", t)
    if m:
        return f"rsi_{m.group(1) or m.group(2)}"
    if "rsi" in t:
        return "rsi_14"
    m = re.search(r"(\d{1,3})[\s-]*(?:day|d)?[\s-]*(sma|ema|moving average|average|ma)\b|(sma|ema)\s*\(?\s*(\d{1,3})", t)
    if m:
        n = m.group(1) or m.group(4)
        kind = (m.group(2) or m.group(3) or "").lower()
        return f"{'ema' if kind == 'ema' else 'sma'}_{n}"
    if "price" in t or "close" in t:
        return "close"
    return None


def _cond(clause: str, last_feat):
    t = clause.lower()
    ops = [(r"crosses? (?:back )?above|crosses? over", "crosses_above"),
           (r"crosses? (?:back )?below|crosses? under", "crosses_below"),
           (r"below|under|less than|drops? (?:below|under)|falls? (?:below|under)|<", "<"),
           (r"above|over|greater than|more than|rises? above|goes above|>", ">")]
    for pat, op in ops:
        m = re.search(pat, t)
        if not m:
            continue
        left = _feat(t[:m.start()]) or last_feat
        rest = t[m.end():]
        right = _feat(rest)
        if right is None or right == left:
            n = re.search(_NUM, rest)
            right = float(n.group(1)) if n else None
        if left and right is not None:
            return {"left": left, "op": op, "right": right}
    return None


def local_parse(text: str):
    t = text.lower().replace(",", " ")
    m = re.search(r"\b(sell|exit|close the position|get out)\b", t)
    buy, sell = (t[:m.start()], t[m.start():]) if m else (t, "")
    entry, exit_ = [], []
    last = None
    for part in re.split(r"\band\b", buy):
        c = _cond(part, last)
        if c:
            entry.append(c)
            last = c["left"]
    for part in re.split(r"\bor\b|\band\b", sell):
        c = _cond(part, last)
        if c:
            exit_.append(c)
            last = c["left"]
    if not entry:
        return None
    if not exit_:
        flip = {">": "<", "<": ">", "crosses_above": "crosses_below", "crosses_below": "crosses_above"}
        exit_ = [{**entry[0], "op": flip[entry[0]["op"]]}]
    return {"name": "Your strategy", "entry": entry, "exit": exit_,
            "explanation": "Parsed from your description by the built-in rule reader."}


def parse_strategy(text: str) -> dict:
    if text in PRESETS:
        return PRESETS[text]
    for name, full in EXAMPLES.items():  # typed just the example's name
        if text.strip().lower() == name.lower():
            return PRESETS[full]
    if not has_key():
        quick = local_parse(text)
        if quick:
            return quick
        raise ValueError("custom strategies need a Gemini API key. Add GEMINI_API_KEY under "
                         "Manage app > Settings > Secrets, or pick an example in the sidebar.")
    try:
        return json.loads(ask_llm(text, SYSTEM, as_json=True))
    except Exception as e:
        quick = local_parse(text)
        if quick:  # AI unavailable or slow: fall back to the built-in reader
            return quick
        if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e) or "UNAUTHENTICATED" in str(e):
            k = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
            raise ValueError(f"Google rejected the API key (it starts with '{k[:3]}' and has {len(k)} "
                             "characters). Use the copy icon in AI Studio to copy the full key.") from None
        raise


def validate(spec: dict) -> None:
    """Never trust model output: every rule must use known features and ops."""
    for key in ("entry", "exit"):
        if not isinstance(spec.get(key), list) or not spec[key]:
            raise ValueError(f"Strategy needs at least one {key} rule.")
        for c in spec[key]:
            if c.get("op") not in OPS:
                raise ValueError(f"Unknown operator: {c.get('op')}")
            for s in (c.get("left"), c.get("right")):
                if isinstance(s, (int, float)):
                    continue
                m = FEAT.match(str(s))
                if not m or (m.group(3) and not 2 <= int(m.group(3)) <= 200):
                    raise ValueError(f"Unknown feature: {s}")


# ---------- Indicators & backtest ----------
def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def feature(df: pd.DataFrame, x) -> pd.Series:
    if isinstance(x, (int, float)):
        return pd.Series(float(x), index=df.index)
    m = FEAT.match(str(x))
    if m.group(1) == "close":
        return df["Close"]
    kind, n = m.group(2), int(m.group(3))
    if kind == "sma":
        return df["Close"].rolling(n).mean()
    if kind == "ema":
        return df["Close"].ewm(span=n, adjust=False).mean()
    return rsi(df["Close"], n)


def condition(df: pd.DataFrame, c: dict) -> pd.Series:
    l, r = feature(df, c["left"]), feature(df, c["right"])
    op = c["op"]
    if op == ">":
        return l > r
    if op == "<":
        return l < r
    if op == "crosses_above":
        return (l > r) & (l.shift(1) <= r.shift(1))
    return (l < r) & (l.shift(1) >= r.shift(1))


def backtest(df: pd.DataFrame, spec: dict, cost_bps: float):
    entry = np.logical_and.reduce([condition(df, c) for c in spec["entry"]])
    exit_ = np.logical_or.reduce([condition(df, c) for c in spec["exit"]])
    pos = pd.Series(np.nan, index=df.index)
    pos[entry] = 1.0
    pos[exit_] = 0.0
    pos = pos.ffill().fillna(0).shift(1).fillna(0)  # act next day: no look-ahead
    ret = df["Close"].pct_change().fillna(0)
    trades = pos.diff().abs().fillna(0)
    strat = pos * ret - trades * cost_bps / 10_000
    return pos, (1 + strat).cumprod(), (1 + ret).cumprod(), strat, ret


def stats(equity: pd.Series, daily: pd.Series, pos: pd.Series) -> dict:
    years = max(len(equity) / 252, 1e-9)
    dd = equity / equity.cummax() - 1
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    return {
        "Total return": f"{equity.iloc[-1] - 1:.1%}",
        "CAGR": f"{equity.iloc[-1] ** (1 / years) - 1:.1%}",
        "Max drawdown": f"{dd.min():.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Trades": int((pos.diff() == 1).sum()),
        "Time in market": f"{pos.mean():.0%}",
    }


@st.cache_data(ttl=3600)
def load(ticker: str, years: int) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    return df[["Close"]].dropna()


# ---------- UI ----------
st.title("📈 StrategyGPT")
st.caption("Describe a trading strategy in plain English. AI turns it into rules and backtests it "
           "on real NSE history. Historical data only, not investment advice.")

with st.sidebar:
    ticker = st.text_input("NSE ticker", "^NSEI", help="e.g. ^NSEI (Nifty 50), RELIANCE.NS, TCS.NS, HDFCBANK.NS")
    years = st.slider("Years of history", 2, 20, 10)
    cost = st.number_input("Cost per trade (basis points)", 0.0, 100.0, 10.0,
                           help="Brokerage + STT + slippage. 10 bps = 0.10%")
    pick = st.selectbox("Try an example", ["(write my own)"] + list(EXAMPLES))

default = EXAMPLES.get(pick, "")
text = st.text_area("Your strategy", default, height=90,
                    placeholder="Buy when RSI 14 is below 30, sell when it goes above 60")

if st.button("Backtest it", type="primary") and text.strip():
    try:
        with st.spinner("Turning your words into rules..."):
            spec = parse_strategy(text.strip())
            validate(spec)
        df = load(ticker.strip(), years)
        if len(df) < 250:
            st.error("Not enough price history for that ticker.")
            st.stop()
    except Exception as e:  # show a friendly error instead of crashing
        st.error(f"Couldn't build that strategy: {e}")
        st.stop()

    pos, eq, bh, strat, ret = backtest(df, spec, cost)
    s, b = stats(eq, strat, pos), stats(bh, ret, pd.Series(1.0, index=df.index))

    st.subheader(spec.get("name", "Your strategy"))
    st.write(spec.get("explanation", ""))
    with st.expander("Rules the AI generated (validated)"):
        st.json({"entry": spec["entry"], "exit": spec["exit"]})

    cols = st.columns(len(s))
    for col, k in zip(cols, s):
        delta = None if k in ("Trades", "Time in market") else f"Buy & hold: {b[k]}"
        col.metric(k, s[k], delta, delta_color="off")

    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq * 100_000, name="Strategy")
    fig.add_scatter(x=bh.index, y=bh * 100_000, name="Buy & hold")
    fig.update_layout(title="Growth of ₹1,00,000", yaxis_title="₹", height=420)
    st.plotly_chart(fig, use_container_width=True)

    def plain_summary() -> str:
        try:
            sr, br = s.get("Total return", "?"), b.get("Total return", "?")
            sd = s.get("Max drawdown", "?")
            return (f"- Strategy return: {sr} vs buy & hold: {br}.\n"
                    f"- Worst fall from a peak (max drawdown): {sd}.\n"
                    "- Costs and slippage can shrink real-world returns.\n"
                    "- Past results don't guarantee future ones; beware of overfitting.")
        except Exception:
            return "Past results don't guarantee future ones; beware of overfitting and costs."

    text = None
    if has_key():
        with st.spinner("Explaining the results..."):
            try:
                text = ask_llm(
                    f"Strategy: {spec}\nTicker: {ticker}, {years} years\nStrategy stats: {s}\nBuy & hold: {b}",
                    "You are a finance teacher for Indian college students. In 4 short bullet points, explain "
                    "these backtest results in plain English: did it beat buy & hold, how risky was it (drawdown), "
                    "and one honest warning (overfitting, costs, past results don't guarantee future ones).",
                    as_json=False)
            except Exception:
                text = None
    st.info(text or plain_summary())
