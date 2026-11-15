from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
ROOT = Path(__file__).resolve().parents[2]

def download_data(ticker: str, start: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, auto_adjust=False, actions=True, progress=False, threads=False)
    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check internet, ticker, and start date.")
    if isinstance(data.columns, pd.MultiIndex):
        levels = [data.columns.get_level_values(i) for i in range(data.columns.nlevels)]
        matching_level = next((i for i, values in enumerate(levels) if ticker in values), None)
        if matching_level is not None:
            data = data.xs(ticker, axis=1, level=matching_level, drop_level=True)
        else:
            known = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
            data.columns = [next((str(part) for part in col if str(part) in known), str(col[0])) for col in data.columns]
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    data.index = pd.to_datetime(data.index)
    if getattr(data.index, "tz", None) is not None: data.index = data.index.tz_localize(None)
    data.index.name = "price_date"
    return data

def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = data.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "price_date" in df.columns:
        df["price_date"] = pd.to_datetime(df["price_date"]); df = df.set_index("price_date")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None: df.index = df.index.tz_localize(None)
    df.index.name = "price_date"
    for col in ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]:
        if col not in df.columns: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df); duplicate_dates = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()
    missing_ohlc = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["open", "high", "low", "close"])
    valid = ((df[["open", "high", "low", "close"]] > 0).all(axis=1) &
             (df["high"] >= df["low"]) &
             (df["high"] >= df[["open", "close"]].max(axis=1)) &
             (df["low"] <= df[["open", "close"]].min(axis=1)))
    invalid_ohlc = int((~valid).sum()); df = df.loc[valid].copy()
    df["volume"] = df["volume"].fillna(0); df["dividends"] = df["dividends"].fillna(0); df["stock_splits"] = df["stock_splits"].fillna(0)
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    report = {"rows_before_cleaning": int(before), "duplicate_dates_removed": duplicate_dates,
              "rows_with_missing_ohlc_removed": missing_ohlc, "invalid_ohlc_rows_removed": invalid_ohlc,
              "rows_after_cleaning": int(len(df)), "remaining_missing_values": int(df.isna().sum().sum())}
    return df, report

def create_features_and_targets(cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = cleaned.copy().sort_index(); close = df["close"].astype(float)
    df["daily_return"] = close.pct_change(); df["log_return"] = np.log(close / close.shift(1))
    for w in [5, 10, 20, 50]: df[f"sma_{w}_ratio"] = close / close.rolling(w, min_periods=w).mean() - 1.0
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean(); ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["ema_12_ratio"] = close / ema12 - 1.0; df["ema_26_ratio"] = close / ema26 - 1.0; df["macd_ratio"] = (ema12 - ema26) / close
    df["volatility_5"] = df["daily_return"].rolling(5, min_periods=5).std(); df["volatility_20"] = df["daily_return"].rolling(20, min_periods=20).std()
    df["high_low_range_pct"] = (df["high"] - df["low"]) / close; df["open_close_return"] = df["close"] / df["open"] - 1.0
    delta = close.diff(); gains = delta.clip(lower=0).rolling(14, min_periods=14).mean(); losses = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan); df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    df.loc[(losses == 0) & (gains > 0), "rsi_14"] = 100.0
    df.loc[(losses == 0) & (gains == 0), "rsi_14"] = 50.0
    df["day_of_week"] = df.index.dayofweek; month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0); df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["target_date"] = pd.Series(df.index, index=df.index).shift(-1)
    df["target_next_day_return"] = close.shift(-1) / close - 1.0
    df["target_next_day_up"] = np.where(df["target_next_day_return"].notna(), (df["target_next_day_return"] > 0).astype(float), np.nan)
    excluded = {"adj_close", "volume", "dividends", "stock_splits", "target_date", "target_next_day_return", "target_next_day_up"}
    features = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    required = features + ["target_next_day_return", "target_next_day_up", "target_date"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy(); df["target_next_day_up"] = df["target_next_day_up"].astype(int)
    return df, features

def chronological_split(data: pd.DataFrame, train_ratio: float = .70, validation_ratio: float = .15):
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1: raise ValueError("Split ratios must be positive and sum to less than 1.")
    df = data.sort_index().copy(); n = len(df)
    if n < 20: raise ValueError(f"Only {n} usable rows after feature engineering; at least 20 are required.")
    a = int(n * train_ratio); b = a + int(n * validation_ratio)

    train = df.iloc[:a].copy()
    validation = df.iloc[a:b].copy()
    test = df.iloc[b:].copy()

    # Purge rows whose next-day target falls into the next partition.
    if len(validation):
        train = train.loc[train["target_date"] < validation.index.min()].copy()
    if len(test):
        validation = validation.loc[validation["target_date"] < test.index.min()].copy()

    return train, validation, test

def make_eda_plots(cleaned: pd.DataFrame, featured: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12,5)); ax.plot(cleaned.index, cleaned.close, lw=1); ax.set(title="NIFTY 50 Closing Price Over Time", xlabel="Trading date", ylabel="Index level"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/"01_closing_price.png", dpi=160); plt.close(fig)
    returns = cleaned.close.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(9,5)); ax.hist(returns, bins=60, edgecolor="black", linewidth=.3); ax.set(title="Distribution of Daily NIFTY 50 Returns", xlabel="Daily return", ylabel="Frequency"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"02_daily_return_distribution.png", dpi=160); plt.close(fig)
    vol = returns.rolling(20).std()
    fig, ax = plt.subplots(figsize=(12,5)); ax.plot(vol.index, vol, lw=.9); ax.set(title="20-Trading-Day Rolling Volatility", xlabel="Trading date", ylabel="Rolling standard deviation"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/"03_rolling_volatility.png", dpi=160); plt.close(fig)
    dd = cleaned.close / cleaned.close.cummax() - 1
    fig, ax = plt.subplots(figsize=(12,4)); ax.fill_between(dd.index, dd.values, 0, alpha=.35); ax.set(title="NIFTY 50 Historical Drawdown", xlabel="Trading date", ylabel="Drawdown"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/"04_drawdown.png", dpi=160); plt.close(fig)
    cols = [c for c in ["daily_return", "sma_5_ratio", "sma_20_ratio", "sma_50_ratio", "volatility_5", "volatility_20", "rsi_14", "macd_ratio", "high_low_range_pct", "open_close_return"] if c in featured.columns]
    corr = featured[cols].corr(); fig, ax = plt.subplots(figsize=(10,8)); im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols))); ax.set_xticklabels(cols, rotation=70, ha="right", fontsize=8); ax.set_yticklabels(cols, fontsize=8); ax.set_title("Feature Correlation Matrix"); fig.colorbar(im, ax=ax, label="Correlation"); fig.tight_layout(); fig.savefig(out/"05_feature_correlation.png", dpi=160); plt.close(fig)
    if (cleaned.volume > 0).any():
        fig, ax = plt.subplots(figsize=(12,4)); ax.plot(cleaned.index, cleaned.volume, lw=.8); ax.set(title="Reported Daily Trading Volume", xlabel="Trading date", ylabel="Volume"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/"06_volume.png", dpi=160); plt.close(fig)

def run(ticker: str, start: str, train_ratio: float, validation_ratio: float, root: Path):
    paths = {"raw":root/"data/raw", "interim":root/"data/interim", "processed":root/"data/processed", "eda":root/"outputs/eda", "datasets":root/"deliverables/datasets"}
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ticker} from {start} ...", flush=True); raw = download_data(ticker, start); raw.to_csv(paths["raw"]/"nifty50_raw.csv")
    cleaned, quality = clean_data(raw); cleaned.to_csv(paths["interim"]/"nifty50_cleaned.csv"); (paths["interim"]/"data_quality_report.json").write_text(json.dumps(quality, indent=2))
    featured, features = create_features_and_targets(cleaned); featured.to_csv(paths["processed"]/"nifty50_features_targets.csv", index_label="price_date")
    train, val, test = chronological_split(featured, train_ratio, validation_ratio)
    parts = [("train",train),("validation",val),("test",test)]; summary=[]
    for name, part in parts:
        part.to_csv(paths["datasets"]/f"{name}.csv", index_label="price_date")
        summary.append({"split":name,"rows":len(part),"first_feature_date":str(part.index.min().date()),"last_feature_date":str(part.index.max().date()),"first_target_date":str(pd.to_datetime(part.target_date).min().date()),"last_target_date":str(pd.to_datetime(part.target_date).max().date())})
    pd.DataFrame(summary).to_csv(paths["processed"]/"split_summary.csv", index=False); pd.DataFrame(summary).to_csv(paths["datasets"]/"split_summary.csv", index=False)
    make_eda_plots(cleaned, featured, paths["eda"])
    (root/"docs/PART1_METHODOLOGY.md").write_text("# Part 1 Methodology\n\nThis pipeline uses Yahoo Finance `^NSEI` as one NIFTY 50 index time series, not 50 separate constituent stocks. It removes duplicate dates, missing OHLC rows and invalid OHLC values. Predictors include daily/log returns, SMA/EMA ratios, MACD ratio, rolling volatility, intraday range, open-close return, RSI-14 and calendar features; volume is retained in the raw and cleaned datasets for inspection and EDA, but excluded from model predictors because reported index volume is inconsistent. Regression target is the next observed trading session close-to-close return. Classification target is 1 for a positive return and 0 for zero/negative return. The time-ordered data is split approximately 70% train, 15% validation and 15% test without shuffling. To prevent target leakage, boundary rows whose next-day target falls into the following partition are purged; two rows are excluded in total. Retain the raw CSV and run summary for reproducibility.\n", encoding="utf-8")
    (root/"docs/DATA_DICTIONARY.md").write_text("# Data Dictionary\n\n`price_date`: predictor date. `target_date`: next observed trading date. `open`, `high`, `low`, `close`: daily index OHLC. `daily_return`: close-to-previous-close return. `log_return`: log return. `sma_*_ratio` and `ema_*_ratio`: close relative to moving average. `volatility_*`: rolling standard deviation of daily returns. `rsi_14`: 14-period RSI. `target_next_day_return`: regression label. `target_next_day_up`: classification label (1 positive, 0 zero/negative).\n\nPredictor columns: " + ", ".join(features) + "\n", encoding="utf-8")
    summary_json = {"ticker":ticker,"start_requested":start,"downloaded_rows":len(raw),"cleaned_rows":len(cleaned),"model_ready_rows":len(featured),"feature_count":len(features),"features":features,"regression_target":"target_next_day_return","classification_target":"target_next_day_up (1=positive, 0=zero/negative)","train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),"data_quality":quality}
    (root/"deliverables/PART1_RUN_SUMMARY.json").write_text(json.dumps(summary_json, indent=2))
    print("\nPART 1 PIPELINE COMPLETED"); print(json.dumps({k:summary_json[k] for k in ["ticker","downloaded_rows","cleaned_rows","model_ready_rows","feature_count","train_rows","validation_rows","test_rows"]}, indent=2))
    print(f"\nDatasets: {paths['datasets']}\nEDA graphs: {paths['eda']}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--ticker",default="^NSEI"); p.add_argument("--start",default="2000-01-01"); p.add_argument("--train-ratio",type=float,default=.70); p.add_argument("--validation-ratio",type=float,default=.15); a=p.parse_args()
    run(a.ticker,a.start,a.train_ratio,a.validation_ratio,ROOT)
if __name__ == "__main__": main()
