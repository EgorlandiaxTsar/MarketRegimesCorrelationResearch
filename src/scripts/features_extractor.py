import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import src.utils as util

DATA_FILE_PATTERN = r"^data_(\d+)_(\d+)\.csv$"


def group_datasets(datasets: list[str]) -> list[tuple[int, int, str]]:
    grouped: list[tuple[int, int, str]] = []
    for path in datasets:
        match = re.match(DATA_FILE_PATTERN, Path(path).name)
        if match: grouped.append((int(match.group(1)), int(match.group(2)), path))
    grouped.sort(key=lambda x: (x[0], x[1]))
    return grouped


def get_dataset(datasets: list[tuple[int, int, str]], group: int, index: int) -> tuple[int, int, str] | None:
    dataset = None
    for e in datasets:
        if e[0] == group and e[1] == index:
            dataset = e
            break
    return dataset


def main():
    parser = argparse.ArgumentParser(
        prog="features_extractor.py",
        description="Sliced dataset features extraction utility script"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Directory from where to extract all CSV datasets (nested directories also)"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory where to store sliced output")
    parser.add_argument("-c", "--candles", type=int, required=True, help="Dataset's candles count")
    parser.add_argument("-f", "--shift", type=int, default=None, help="Candles shift between datasets. Defaults to defined candles count. Cannot be greater than candles count, otherwise, it will be set to default")
    args = parser.parse_args()
    if not util.is_valid_location(args.data) or not util.is_valid_location(args.output):
        print("Incorrect input/output file locations")
        return
    shift = args.candles if args.shift is None else min(args.candles, args.shift) 
    dataset_shift = args.candles / shift
    datasets = group_datasets([str(p) for p in Path(args.data).rglob("*.csv")])
    for dataset in datasets:
        if dataset[1] < dataset_shift: continue
        prev_dataset = get_dataset(datasets, dataset[0], dataset[1] - dataset_shift)
        if prev_dataset is None: continue
        df = pd.read_csv(prev_dataset[2])
        df["ma15"], df["rsi15"], df["atr60"] = util.ma(df, 15), util.rsi(df, 15), util.ma(df, 60)
        df["candle_spread"], df["candle_wb_spread"] = util.candle_spread_pct(df), util.candle_wick_spread_pct(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        with open(f"{args.output}/features_{dataset[0]}_{dataset[1]}.json", "w") as f:
            json.dump(
                {
                    "metadata": {
                        "input": prev_dataset[2],  # Dataset with which all metrics are computed (n - 1)
                        "target": dataset[2]  # Dataset for testing using computed metrics (n)
                    },
                    "features": {
                        "scalar": {
                            "std": util.std(df),
                            "rstd": util.rstd(df),
                            "ma_distance_ratio": util.ma_distance_ratio(df, ma="ma15"),
                            "rsi_crosses": util.rsi_crosses_count(df, "rsi15", 20, 80),
                            "entropy": util.shannon_entropy(df, bins=20),
                            "regression_slope": util.regression_slope(df),
                            "regression_sqr": util.regression_r2(df),
                            "volatility_clustering": util.volatility_clustering(df, lag=1),
                            "max_drawdown_ratio": util.max_drawdown_ratio(df),
                            "time_to_trough_ratio": util.time_to_trough_ratio(df),
                            "recovery_ratio": util.recovery_ratio(df),
                            "path_length_ratio": util.path_length_ratio(df),
                            "sign_change_ratio": util.sign_change_ratio(df),
                            "drawdown_convexity": util.drawdown_convexity(df),
                            "drop_velocity": util.drop_velocity(df)
                        },
                        "graphical": {
                            "ma15": df["ma15"].to_list(),
                            "rsi15": df["rsi15"].to_list(),
                            "atr60": df["atr60"].to_list()
                        }
                    }
                },
                f
            )


if __name__ == "__main__":
    main()
