# streamlit_btc_alt_predictor_v4_investment_ready.py
# Upgraded predictor for portfolio allocation research
# - Adds multi-feature models (rolling returns, volatility, alt/BTC ratio)
# - Option to use RandomForestRegressor (non-linear) or SGD/Linear
# - Accuracy trend chart (history of past prediction errors)
# - Keeps self-learning from older predictions (>3 days)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta

# ------------------------- Config -------------------------
APP_NAME = "BTC → Alts Predictor v4 (Investment-ready)"
DATA_DIR = "models_data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
PRED_CSV = os.path.join(DATA_DIR, "predictions.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")
BTC = "BTC-USD"
DEFAULT_ALTS = ["ETH-USD", "SOL-USD", "BNB-USD"]
LEARN_DELAY = 3  # days
LOOKBACK_DAYS = 365
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------- Utilities -------------------------

def safe_download(tickers, period="1y", interval="1d", start=None, end=None):
    # handle single ticker vs list and multiindex
    if isinstance(tickers, (list, tuple)) and len(tickers) == 1:
        tickers = tickers[0]
    df = yf.download(tickers, period=period, interval=interval, start=start, end=end, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # prefer Adj Close
        if 'Adj Close' in df.columns.get_level_values(0):
            out = df['Adj Close']
        else:
            out = df['Close']
    else:
        if 'Adj Close' in df.columns:
            out = df[['Adj Close']]
            # if single ticker, ensure column named
            if isinstance(tickers, str):
                out.columns = [tickers]
        elif 'Close' in df.columns:
            out = df[['Close']]
            if isinstance(tickers, str):
                out.columns = [tickers]
        else:
            return pd.DataFrame()
    if isinstance(out, pd.Series):
        out = out.to_frame()
    return out.dropna(how='all')


def ensure_pred_file():
    if not os.path.exists(PRED_CSV):
        cols = ['date','btc_pct','ticker','pred_return','pred_price','actual_return']
        pd.DataFrame(columns=cols).to_csv(PRED_CSV, index=False)


def load_predictions():
    ensure_pred_file()
    return pd.read_csv(PRED_CSV, parse_dates=['date'])


def save_prediction_rows(rows):
    ensure_pred_file()
    df_new = pd.DataFrame(rows)
    if os.path.exists(PRED_CSV):
        df_old = pd.read_csv(PRED_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(PRED_CSV, index=False)


def model_file(ticker, model_name):
    safe = ticker.replace('/', '_')
    return os.path.join(MODELS_DIR, f"{safe}_{model_name}.joblib")


def load_model(ticker, model_name):
    p = model_file(ticker, model_name)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def save_model(m, ticker, model_name):
    joblib.dump(m, model_file(ticker, model_name))

# ------------------------- Feature engineering -------------------------

def build_features(btc_series, asset_series, window=7):
    # both are pandas Series indexed by date
    df = pd.concat([btc_series, asset_series], axis=1, join='inner')
    df.columns = ['btc','asset']
    # returns
    df['btc_ret_1d'] = df['btc'].pct_change()
    df['asset_ret_1d'] = df['asset'].pct_change()
    df['btc_ret_7d'] = df['btc'].pct_change(periods=window)
    df['asset_ret_7d'] = df['asset'].pct_change(periods=window)
    # volatility (rolling std of daily returns)
    df['btc_vol_7d'] = df['btc_ret_1d'].rolling(window).std()
    # ratio (asset price relative to btc)
    df['asset_btc_ratio'] = df['asset'] / df['btc']
    # momentum: difference between 7d and 1d
    df['btc_mom'] = df['btc_ret_7d'] - df['btc_ret_1d']
    # drop na
    df = df.dropna()
    return df

# ------------------------- App UI -------------------------

st.set_page_config(page_title=APP_NAME, layout='wide')
st.title(APP_NAME)
st.caption('Use this as research tooling — not financial advice.')

# Accuracy metric and trend
preds_df = load_predictions()
if {'pred_return','actual_return'}.issubset(preds_df.columns) and not preds_df[['pred_return','actual_return']].dropna().empty:
    preds_df['error_abs'] = (preds_df['actual_return'] - preds_df['pred_return']).abs()
    mean_err = preds_df['error_abs'].mean()
    accuracy = max(0.0, 100.0 - mean_err*100.0)
    color = '🟢' if accuracy>=70 else ('🟡' if accuracy>=40 else '🔴')
    st.metric('Model Accuracy', f"{color} {accuracy:.2f}%")
    # trend chart: rolling accuracy (7-day)
    preds_df = preds_df.sort_values('date')
    rolling_err = preds_df['error_abs'].rolling(window=7, min_periods=1).mean()
    rolling_acc = 100 - rolling_err*100
    st.line_chart(pd.DataFrame({'accuracy': rolling_acc.values}, index=preds_df['date']))
else:
    st.metric('Model Accuracy', 'N/A', 'insufficient data')

# Sidebar controls
with st.sidebar:
    st.header('Controls')
    alts_input = st.text_area('Alt tickers (comma separated)', value=', '.join(DEFAULT_ALTS))
    alts = [t.strip().upper() for t in alts_input.split(',') if t.strip()]
    lookback = st.selectbox('Historical lookback', options=['6mo','1y','2y'], index=1)
    model_choice = st.selectbox('Model', options=['sgd','linear','rf'], index=2)
    retrain_days = st.number_input('Retrain recent N days', min_value=30, max_value=3650, value=365)
    window_days = st.slider('Feature window days (for 7d features)', min_value=3, max_value=30, value=7)
    st.markdown('---')
    st.write('Data dir:'); st.write(DATA_DIR)

# fetch BTC current
with st.spinner('Fetching BTC...'):
    btc_df = safe_download(BTC, period='7d')
    if btc_df.empty:
        st.error('Could not fetch BTC price')
        st.stop()
    btc_price = float(btc_df.iloc[-1,0])

st.write(f'BTC current: ${btc_price:,.2f}')

# user target
col1, col2 = st.columns([2,1])
with col1:
    mode = st.radio('Target input type', ['Absolute price','% change'], index=0)
    if mode=='Absolute price':
        btc_target = st.number_input('BTC target price', value=round(btc_price*1.05,2))
        btc_pct = (btc_target - btc_price)/btc_price
    else:
        btc_pct = st.number_input('BTC % (e.g. 5 for 5%)', value=5.0)/100.0
        btc_target = btc_price*(1+btc_pct)
    st.write(f'Target ${btc_target:,.2f} ({btc_pct*100:+.2f}%)')

with col2:
    predict = st.button('Predict')
    
# =============================================================
# Step: Historical Bootstrapping for Initial Learning
# =============================================================
st.sidebar.info("📈 Bootstrapping models with past BTC/Altcoin data...")

try:
    hist_days = 90  # adjust this for how much history you want
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=hist_days)

    btc_hist = yf.download(BTC, start=start_date, end=end_date, progress=False)
    if "Adj Close" not in btc_hist.columns and "Close" in btc_hist.columns:
        btc_hist["Adj Close"] = btc_hist["Close"]
    btc_hist["btc_ret"] = btc_hist["Adj Close"].pct_change() * 100

    for alt in alts:
        alt_hist = yf.download(alt, start=start_date, end=end_date, progress=False)
        if "Adj Close" not in alt_hist.columns and "Close" in alt_hist.columns:
            alt_hist["Adj Close"] = alt_hist["Close"]
        alt_hist["alt_ret"] = alt_hist["Adj Close"].pct_change() * 100

        merged = pd.merge(
            btc_hist[["btc_ret"]],
            alt_hist[["alt_ret"]],
            left_index=True,
            right_index=True,
        ).dropna()

        if not merged.empty:
            X = merged[["btc_ret"]].values
            y = merged["alt_ret"].values

            model_name = model_choice
            model = load_model(alt, model_name)
            if model is None:
                if model_choice == "sgd":
                    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
                elif model_choice == "linear":
                    model = make_pipeline(StandardScaler(), LinearRegression())
                else:
                    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)

            model.fit(X, y)
            save_model(model, alt, model_name)
            st.sidebar.success(f"{alt} trained on {len(merged)} days of real data.")
        else:
            st.sidebar.warning(f"No overlapping data found for {alt}.")
except Exception as e:
    st.sidebar.error(f"Bootstrapping error: {e}")

# ------------------------- Predict & Train -------------------------
if predict:
    rows_to_log = []
    results = []
    for alt in alts:
        # fetch history
        hist = safe_download([BTC, alt], period='1y')
        if hist.empty or BTC not in hist.columns or alt not in hist.columns:
            st.warning(f'No data for {alt}'); continue
        btc_s = hist[BTC]
        alt_s = hist[alt]
        feat = build_features(btc_s, alt_s, window=window_days)
        if feat.empty:
            st.warning(f'Not enough history for {alt}'); continue
        # features X and target y (we use 7d asset return as target)
        X = feat[['btc_ret_1d','btc_ret_7d','btc_vol_7d','asset_btc_ratio','btc_mom']].values
        y = feat['asset_ret_7d'].values
        # select model
        model_name = model_choice
        model = load_model(alt, model_name)
        if model is None:
            if model_choice=='sgd':
                model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
            elif model_choice=='linear':
                model = make_pipeline(StandardScaler(), LinearRegression())
            else:
                model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        # fit
        try:
            model.fit(X,y)
            save_model(model, alt, model_name)
        except Exception as e:
            st.warning(f'Could not fit model for {alt}: {e}')
            continue
        # prepare current feature vector based on latest prices
        latest = feat.iloc[-1:]
        X_now = latest[['btc_ret_1d','btc_ret_7d','btc_vol_7d','asset_btc_ratio','btc_mom']].values
        # predict 7d return (decimal)
        try:
            pred_ret = float(model.predict(X_now)[0])
        except Exception:
            pred_ret = 0.0
        # predicted price after 7 days
        cur_price = float(alt_s.iloc[-1])
        pred_price = cur_price * (1 + pred_ret)
        results.append((alt, pred_ret*100, cur_price, pred_price))
        rows_to_log.append({'date': datetime.utcnow().strftime('%Y-%m-%d'), 'btc_pct': btc_pct, 'ticker': alt, 'pred_return': pred_ret, 'pred_price': pred_price, 'actual_return': np.nan})
    # display
    if results:
        st.subheader('Predictions (7-day horizon):')
        for alt, pct, cur, p in results:
            st.write(f"**{alt}** → Predicted 7d move: {pct:+.2f}% | Current: ${cur:,.2f} | Target: ${p:,.2f}")
        save_prediction_rows(rows_to_log)
        st.success('Predictions saved and models updated.')
    else:
        st.info('No predictions produced.')

# ------------------------- Self-learning on startup -------------------------
# Find predictions older than LEARN_DELAY days and fill actuals, then do incremental updates
preds = load_predictions()
if not preds.empty and 'actual_return' in preds.columns:
    # Handle mixed date formats safely (ISO or plain date)
    preds['date'] = pd.to_datetime(preds['date'], format='mixed', errors='coerce')

    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=LEARN_DELAY))
    to_learn = preds[(preds['actual_return'].isna()) & (preds['date'] <= cutoff)]

    if not to_learn.empty:
        summary = {}
        for idx, row in to_learn.iterrows():
            alt = row['ticker'] if 'ticker' in row else row.get('alt')
            check_date = (pd.to_datetime(row['date']) + timedelta(days=LEARN_DELAY)).date()
            price, time = safe_download(
                alt,
                period='7d',
                start=check_date.strftime('%Y-%m-%d'),
                end=(check_date + timedelta(days=1)).strftime('%Y-%m-%d')
            )

            # safe_download returns df; adapt
            actual_price = None
            if isinstance(price, pd.DataFrame) and not price.empty:
                actual_price = float(price.iloc[-1, 0])
            if actual_price is None:
                continue

            base_price = row['pred_price'] / (1 + row['pred_return']) if row['pred_return'] != 0 else row['pred_price']
            actual_ret = (actual_price - base_price) / base_price
            preds.loc[idx, 'actual_return'] = actual_ret

            # incremental update: load model and partial_fit if supported
            model = load_model(alt, model_choice)
            if model is not None:
                try:
                    # extract scaler + regressor if pipeline
                    if hasattr(model, 'named_steps'):
                        scaler = model.named_steps[list(model.named_steps.keys())[0]]
                        reg = model.named_steps[list(model.named_steps.keys())[-1]]
                        X_new = np.array([[row['btc_pct']]])
                        Xs = scaler.transform(X_new)
                        if hasattr(reg, 'partial_fit'):
                            reg.partial_fit(Xs, np.array([actual_ret]))
                            save_model(model, alt, model_choice)
                except Exception:
                    pass

        preds.to_csv(PRED_CSV, index=False)

# ------------------------- Footer -------------------------
st.markdown('---')
st.write('This tool is for research and portfolio planning. Do not use without risk management.')
