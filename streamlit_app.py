# streamlit_btc_alt_predictor.py
# Streamlit app to predict altcoin price moves given a BTC target using historical correlations.
# Features:
# - Fetches historical prices with yfinance
# - Trains per-alt incremental models (SGDRegressor) mapping BTC returns -> alt returns
# - Lets you enter a BTC target price (absolute or %), shows predicted % move and predicted price for selected alts
# - Stores history and allows "record actuals" to update training data (online learning via partial_fit)
# - Saves models and history to disk (models/*.joblib, data/history.csv)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta

# --- Constants / defaults ---
DEFAULT_ALTS = ["ETH-USD", "SOL-USD", "BNB-USD"]
BTC_TICKER = "BTC-USD"
DATA_DIR = "models_data"
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")
MODELS_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# --- Utility functions ---

def fetch_prices(tickers, period="1y", interval="1d"):
    # Use yfinance to download adjusted close
    df = yf.download(tickers, period=period, interval=interval, progress=False, threads=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    return df


def pct_return(series):
    return series.pct_change().dropna()


def ensure_history():
    if not os.path.exists(HISTORY_CSV):
        df = pd.DataFrame(columns=["date","ticker","btc_price","asset_price","btc_return","asset_return"])
        df.to_csv(HISTORY_CSV, index=False)


def load_history():
    ensure_history()
    return pd.read_csv(HISTORY_CSV, parse_dates=["date"]) if os.path.exists(HISTORY_CSV) else pd.DataFrame()


def append_history(new_rows: pd.DataFrame):
    ensure_history()
    new_rows.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)


def model_path(ticker):
    safe = ticker.replace('/', '_')
    return os.path.join(MODELS_DIR, f"model_{safe}.joblib")


def load_or_create_model(ticker, model_type="sgd"):
    p = model_path(ticker)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception:
            pass
    # create new incremental model pipeline: scaler + SGDRegressor
    if model_type == "sgd":
        model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    else:
        # fallback to simple LinearRegression (not incremental)
        model = make_pipeline(StandardScaler(), LinearRegression())
    return model


def save_model(ticker, model):
    joblib.dump(model, model_path(ticker))

# --- Streamlit UI ---

st.set_page_config(page_title="BTC -> Alts Predictor", layout="wide")
st.title("BTC → Altcoins Quick Predictor")
st.write("Enter a BTC target or % change and see predicted % and prices for selected altcoins based on historical correlation.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    alts_input = st.text_area("Alt tickers (comma separated)", value=", ".join(DEFAULT_ALTS))
    alts = [t.strip().upper() for t in alts_input.split(",") if t.strip()]
    period = st.selectbox("Historical lookback", options=["6mo","1y","2y","5y"], index=1)
    model_type = st.selectbox("Model type", options=["sgd","linear"], index=0)
    retrain_days = st.number_input("Retrain using last N days of data", min_value=30, max_value=3650, value=365)
    st.markdown("---")
    st.markdown("**Data & model controls**")
    if st.button("Refresh historical data & retrain all"):
        st.session_state.get("retrain_all", False)
        st.session_state["retrain_all"] = True

# Main: fetch current BTC price to compute target by %
with st.spinner("Fetching latest BTC price..."):
    recent = yf.download([BTC_TICKER], period="7d", interval="1d", progress=False, threads=False)["Adj Close"].dropna()
    if recent.empty:
        st.error("Could not fetch current BTC price. Check network or ticker.")
        st.stop()
    current_btc = float(recent.iloc[-1].values[0])

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Enter BTC target")
    btc_target_mode = st.radio("Target input type", ["Absolute price","Percentage change from current BTC"], index=0)
    if btc_target_mode == "Absolute price":
        btc_target = st.number_input("BTC target price (USD)", value=round(current_btc*1.05,2))
        btc_pct = (btc_target - current_btc) / current_btc
    else:
        btc_pct_input = st.number_input("BTC % change (e.g. 10 for +10%, -5 for -5%)", value=5.0)
        btc_pct = float(btc_pct_input)/100.0
        btc_target = current_btc * (1 + btc_pct)
    st.write(f"Current BTC: ${current_btc:,.2f} → Target ${btc_target:,.2f} ({btc_pct*100:+.2f}%)")

with col2:
    st.subheader("Quick actions")
    if st.button("Predict now"):
        st.session_state["do_predict"] = True
    st.markdown("---")
    st.write("Model notes:")
    st.write("- Models learn from historical BTC vs alt returns.")
    st.write("- You can record actual outcomes later to let the models learn from mistakes (online update).")

# Predict block
if st.session_state.get("do_predict", False) or st.session_state.get("retrain_all", False):
    with st.spinner("Fetching historical data and training models..."):
        tickers = [BTC_TICKER] + alts
        prices = fetch_prices(tickers, period=period)
        if BTC_TICKER not in prices.columns:
            st.error("BTC price missing from fetched data. Try a different lookback or check network.")
            st.stop()

        btc_prices = prices[BTC_TICKER].dropna()
        btc_returns = pct_return(btc_prices)

        results = []
        models = {}
        for alt in alts:
            if alt not in prices.columns:
                st.warning(f"Ticker {alt} not found in yfinance data; skipping")
                continue
            alt_prices = prices[alt].dropna()
            # align dates
            df = pd.concat([btc_prices, alt_prices], axis=1, join='inner').dropna()
            df.columns = ["btc","asset"]
            df_ret = df.pct_change().dropna()
            # use last N days if requested
            if retrain_days and retrain_days < len(df_ret):
                df_ret = df_ret.tail(retrain_days)

            X = df_ret[["btc"]].values
            y = df_ret["asset"].values

            model = load_or_create_model(alt, model_type=model_type)
            try:
                # For SGDRegressor pipeline, we can call fit (cold start). If it's LinearRegression, fit.
                model.fit(X, y)
            except Exception as e:
                st.warning(f"Could not fit model for {alt}: {e}")
                continue
            save_model(alt, model)
            models[alt] = model

            # Predict alt return corresponding to btc_pct
            X_pred = np.array([[btc_pct]])
            pred_ret = float(model.predict(X_pred)[0])
            current_price = float(alt_prices.iloc[-1])
            pred_price = current_price * (1 + pred_ret)
            results.append({"ticker": alt, "current": current_price, "predicted_price": pred_price, "predicted_return": pred_ret})

        # show results
        if results:
            res_df = pd.DataFrame(results)
            res_df["current"] = res_df["current"].map(lambda x: f"${x:,.4f}")
            res_df["predicted_price"] = res_df["predicted_price"].map(lambda x: f"${x:,.4f}")
            res_df["predicted_return_pct"] = (res_df["predicted_return"]*100).map(lambda x: f"{x:+.2f}%")
            st.subheader("Predictions")
            st.table(res_df[["ticker","current","predicted_price","predicted_return_pct"]])

            st.markdown("---")
            st.write("You can record actual outcomes (future) below to update the models with real observed errors — this enables the model to learn from mistakes incrementally.")

            # Save a snapshot to history for record (optional)
            snapshot = []
            today = datetime.utcnow().date().isoformat()
            for r in results:
                snapshot.append({"date": today, "ticker": r["ticker"], "btc_price": current_btc, "asset_price": float(r["current"].replace('$','').replace(',','')), "btc_return": btc_pct, "asset_return": r["predicted_return"]})
            # append to history as 'prediction' records (we keep predictions too)
            snapshot_df = pd.DataFrame(snapshot)
            # Tagging predictions optional — keep same columns
            append_history(snapshot_df)

            # Show option to record actuals now
            st.subheader("Record actual outcomes (when known)")
            with st.form("record_actuals"):
                st.write("Enter actual BTC price at target time and actual prices for alts (or leave blank to skip). The app will compute the actual returns and update models.")
                actual_btc = st.number_input("Actual BTC price at time of outcome (USD)", value=float(btc_target))
                actuals = {}
                for alt in alts:
                    actuals[alt] = st.text_input(f"Actual price for {alt} (leave blank if unknown)", value="")
                submitted = st.form_submit_button("Record & Update Models")
                if submitted:
                    # collect rows and retrain with partial_fit (if using SGD)
                    rows = []
                    for alt, val in actuals.items():
                        if val.strip() == "":
                            continue
                        try:
                            valf = float(val)
                        except:
                            st.warning(f"Invalid price for {alt}: {val}")
                            continue
                        btc_ret_actual = (actual_btc - current_btc) / current_btc
                        asset_ret_actual = (valf - float(prices[alt].iloc[-1])) / float(prices[alt].iloc[-1])
                        rows.append({"date": datetime.utcnow().isoformat(), "ticker": alt, "btc_price": actual_btc, "asset_price": valf, "btc_return": btc_ret_actual, "asset_return": asset_ret_actual})

                    if rows:
                        rows_df = pd.DataFrame(rows)
                        append_history(rows_df)
                        # Try partial_fit if model is SGD
                        for row in rows:
                            alt = row["ticker"]
                            model = load_or_create_model(alt, model_type=model_type)
                            X_new = np.array([[row["btc_return"]]])
                            y_new = np.array([row["asset_return"]])
                            try:
                                # only SGDRegressor pipeline supports partial_fit on the final estimator
                                if hasattr(model.named_steps[list(model.named_steps.keys())[-1]], "partial_fit"):
                                    # extract scaler and regressor
                                    scaler = model.named_steps[list(model.named_steps.keys())[0]]
                                    reg = model.named_steps[list(model.named_steps.keys())[-1]]
                                    Xs = scaler.transform(X_new)
                                    reg.partial_fit(Xs, y_new)
                                    save_model(alt, model)
                                else:
                                    # full retrain from history
                                    st.info(f"Model for {alt} does not support partial_fit — retraining from history instead.")
                                    # retrain from entire saved history
                                    hist = load_history()
                                    h = hist[hist.ticker==alt]
                                    if len(h) > 5:
                                        X = h[["btc_return"]].values
                                        y = h["asset_return"].values
                                        model.fit(X,y)
                                        save_model(alt, model)
                            except Exception as e:
                                st.warning(f"Could not update model for {alt}: {e}")
                        st.success("Recorded actuals and updated models (if supported).")
                    else:
                        st.info("No actuals provided.")

        else:
            st.info("No predictions were produced (no valid alt tickers).")

# Provide guidance and warnings
st.markdown("---")
st.header("Important notes & limitations")
st.write("This tool is a *simple* correlation-based predictor. A few important cautions:")
st.write("- Historical correlation is not a guarantee of future behavior. Crypto markets are volatile and relationships change quickly.")
st.write("- The model above uses BTC -> alt return regressions; it ignores many important drivers (liquidity, news, sentiment, macro, etc.).")
st.write("- You should NOT rely on this for high-leverage or large financial decisions. Consider this a research/assistant tool, not trading advice.")
st.write("- 100% accuracy is impossible to guarantee; models can improve but will never be perfect. Use risk management and do your own due diligence.")

st.markdown("---")
st.write("If you'd like, I can:")
st.write("1. Expand the model to use lagged returns, volatility, and multiple features (ETH returns, market cap) — typically improves performance.")
st.write("2. Add cross-validation and simple performance metrics (MAE, RMSE) and show backtests on historical BTC moves.")
st.write("3. Package this as a Docker image or a one-click Streamlit share app.")

st.write("Tell me which of the above you'd like next, or if you want the app modified to your exact tickers/timeframes.")

