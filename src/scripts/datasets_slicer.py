import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import src.utils as util


def main():
    parser = argparse.ArgumentParser(prog="datasets_slicer.py", description="Raw dataset slicer utility script")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Directory from where to extract all CSV datasets (nested directories also)"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory where to store sliced output")
    parser.add_argument("-s", "--step", type=int, required=True, help="Sliced dataset target size")
    parser.add_argument("-f", "--shift", type=int, default=None, help="Candles shift after each iteration. Defaults to defined steps count. Cannot be greater than steps count, otherwise, it will be set to default")
    args = parser.parse_args()
    if not util.is_valid_location(args.data) or not util.is_valid_location(args.output):
        print("Incorrect input/output file locations")
        return
    shift = args.step if args.shift is None else min(args.step, args.shift) 
    file_index, dataset_index = 0, 0
    dataset_files: list[str] = [str(p) for p in Path(args.data).rglob("*.csv")]
    for dataset_location in dataset_files:
        df = pd.read_csv(dataset_location)
        df = df.rename(columns={"timestamp": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        df = df[(df != 0).all(axis=1)]
        index = args.step
        if index >= len(df):
            continue
        while index < len(df):
            pd.DataFrame(df.iloc[index - args.step:index]).to_csv(
                f"{args.output}/data_{dataset_index}_{file_index}.csv",
                index=False
            )
            index += shift
            file_index += 1
        file_index = 0
        dataset_index += 1


if __name__ == "__main__":
    main()
