# Data Dictionary

`price_date`: predictor date. `target_date`: next observed trading date. `open`, `high`, `low`, `close`: daily index OHLC. `daily_return`: close-to-previous-close return. `log_return`: log return. `sma_*_ratio` and `ema_*_ratio`: close relative to moving average. `volatility_*`: rolling standard deviation of daily returns. `rsi_14`: 14-period RSI. `target_next_day_return`: regression label. `target_next_day_up`: classification label (1 positive, 0 zero/negative).

Predictor columns: close, high, low, open, daily_return, log_return, sma_5_ratio, sma_10_ratio, sma_20_ratio, sma_50_ratio, ema_12_ratio, ema_26_ratio, macd_ratio, volatility_5, volatility_20, high_low_range_pct, open_close_return, rsi_14, day_of_week, month_sin, month_cos
