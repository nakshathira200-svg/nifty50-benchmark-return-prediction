import numpy as np
import pandas as pd
from src.data.part1_pipeline import clean_data, create_features_and_targets, chronological_split

def sample_data(n=180):
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 10000 + np.arange(n)*4 + np.sin(np.arange(n)/4)*20
    return pd.DataFrame({"Open":close-5,"High":close+10,"Low":close-10,"Close":close,"Volume":np.arange(n)+100,"Dividends":0,"Stock Splits":0},index=idx)

def test_cleaning_removes_duplicates_and_invalid_ohlc():
    data=sample_data(80); data=pd.concat([data,data.iloc[[-1]]]); data.iloc[0,data.columns.get_loc("High")]=1
    clean, report=clean_data(data)
    assert not clean.index.has_duplicates
    assert report["duplicate_dates_removed"] == 1
    assert report["invalid_ohlc_rows_removed"] == 1

def test_target_is_next_observation_return():
    clean,_=clean_data(sample_data(100)); feat,_=create_features_and_targets(clean); row=feat.iloc[0]; nxt=feat.index[1]
    expected=clean.loc[nxt,"close"]/clean.loc[row.name,"close"]-1
    assert row["target_date"] == nxt
    assert abs(row["target_next_day_return"]-expected)<1e-12

def test_chronological_split_is_ordered_and_complete():
    clean, _ = clean_data(sample_data(180))
    feat, _ = create_features_and_targets(clean)
    train, val, test = chronological_split(feat)

    assert train.index.max() < val.index.min() < test.index.min()

    # Two boundary rows are intentionally purged.
    assert len(train) + len(val) + len(test) == len(feat) - 2

    # No next-day target crosses into the next partition.
    assert train["target_date"].max() < val.index.min()
    assert val["target_date"].max() < test.index.min()
