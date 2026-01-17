import argparse
import math
import time
import threading
from datetime import datetime

import pandas as pd
import requests

import src.utils as util
from src.env.data import Candle

SHUTDOWN = threading.Event()
MORALIS_URL = "https://solana-gateway.moralis.io/token/mainnet/pairs"
TIMEFRAME_SECONDS = {
    "1s": 1,
    "10s": 10,
    "30s": 30,
    "1min": 60,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}


def stream_fetch(
        pair_addr: str,
        start: str,
        on_completed: callable,
        end: str | None,
        api_key: str,
        timeframe: str = "1h",
        currency: str = "usd",
        on_error: callable = None
):
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    step_seconds = TIMEFRAME_SECONDS[timeframe]
    candles = []
    headers = {"accept": "application/json", "X-API-Key": api_key}
    start_timestamp = to_unix(start)
    alive, error, batch = True, False, 0
    try: 
        if end is None:
            while alive:
                alive, error, batch = _process_data_batch(
                    pair_addr,
                    timeframe,
                    currency,
                    headers,
                    batch,
                    start_timestamp,
                    step_seconds,
                    candles,
                    on_error
                )
                if error: candles = []
        else:
            end_timestamp = to_unix(end)
            total_candles = max(0, (end_timestamp - start_timestamp) // step_seconds)
            if total_candles == 0:
                on_completed([])
                return
            batches = math.ceil(total_candles / 1000)
            while alive:
                alive, error, batch = _process_data_batch(
                    pair_addr,
                    timeframe,
                    currency,
                    headers,
                    batch,
                    start_timestamp,
                    step_seconds,
                    candles,
                    on_error,
                    batches,
                    end_timestamp
                )
                if error: candles = []
    except KeyboardInterrupt:
        print("Shutown requested (CTRL+C), completing last request and flushing accumulated data batch...")
        SHUTDOWN.set()
    finally:
        on_completed(_process_candles(candles.copy()))


def _fetch_data_batch(
        pair_addr: str,
        batch_start: int,
        batch_end: int,
        timeframe: str,
        currency: str,
        headers: dict,
        max_retries: int = 1
) -> list[Candle]:
    candles = []
    retries = 0
    while retries < max_retries:
        res = requests.get(
            f"{MORALIS_URL}/{pair_addr}/ohlcv",
            headers=headers,
            params={
                "fromDate": batch_start,
                "toDate": batch_end,
                "timeframe": timeframe,
                "currency": currency,
                "limit": 1000
            }
        )
        if res.status_code == 200:
            data = res.json().get("result", [])
            for e in data: candles.append(
                Candle(e["open"], e["high"], e["low"], e["close"], e["volume"], e["timestamp"])
            )
            break
        elif res.status_code == 500:
            retries += 1
            print(f"Moralis 500 error. Retrying {retries}/{max_retries} in 10s...")
            time.sleep(10)
        else:
            raise RuntimeError(f"Moralis API error: {res.status_code} | {res.text}")
    else:
        raise RuntimeError(f"Failed to fetch data from Moralis after {max_retries} retries")
    return candles


def _process_data_batch(
        pair_addr: str,
        timeframe: str,
        currency: str,
        headers: dict[str, any],
        batch: int,
        start: float | int,
        step: int,
        accumulator: list[Candle] | None = None,
        on_error: callable = None,
        max_batches: int | None = None,
        end: float | int | None = None
) -> tuple[bool, bool, int]:
    def alive() -> bool:
        return batch < max_batches if max_batches is not None else True

    if SHUTDOWN.is_set(): return False, False, batch
    batch_str = " (streaming)" if max_batches is None else f"/{max_batches}"
    accumulated_str = "" if accumulator is None else f", accumulated {len(accumulator)} candles"
    time_str = "" if accumulator is None or len(accumulator) == 0 else f", time index {accumulator[-1].timestamp}"
    print(f"Fetching batch {batch + 1}{batch_str}{accumulated_str}{time_str}...")
    batch_start = start + batch * 1000 * step
    batch_end = min(batch_start + 1000 * step, end) if end is not None else batch_start + 1000 * step
    try:
        candles = _fetch_data_batch(pair_addr, batch_start, batch_end, timeframe, currency, headers, max_retries=1)
        if accumulator is not None: accumulator += candles
        batch += 1
    except Exception as e:
        batch += 1
        if on_error:
            on_error(e, _process_candles(accumulator.copy() if accumulator is not None else candles.copy()))
            return alive(), True, batch
        else:
            return False, True, batch
    if len(candles) == 0:
        print(f"No more candles — stopping at batch {batch + 1}.")
        return False, False, batch
    return alive(), False, batch


def _process_candles(candles: list[Candle]) -> list[Candle]:
    candles = candles.copy()
    candles.sort(key=lambda x: x.timestamp)
    unique = []
    seen = set()
    for c in candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    return unique


def write(data: list[Candle], location: str):
    df = pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in data],
            "open": [c.open for c in data],
            "high": [c.high for c in data],
            "low": [c.low for c in data],
            "close": [c.close for c in data],
            "volume": [c.volume for c in data]
        }
    )
    df.to_csv(location, index=False)
    print(f"Saved {len(data)} candles → {location}")


def to_unix(t):
    return int(t) if t.isdigit() else int(datetime.fromisoformat(t).timestamp())


def main():
    parser = argparse.ArgumentParser(
        prog="moralis_fetcher.py",
        description="Solana memecoins candlestick data fetcher script"
    )
    parser.add_argument("-a", "--addr", type=str, required=True, help="Pair address to query (NOT token address!).")
    parser.add_argument(
        "-s",
        "--start",
        type=str,
        required=True,
        help="Point in time from where to fetch the data. Must be in ISO8086 format (2025-11-22T00:15:00Z)"
    )
    parser.add_argument(
        "-e",
        "--end",
        type=str,
        default=None,
        help="Point in time to where to fetch the data. Must be in ISO8086 format (2025-11-29T00:15:00Z)"
    )
    parser.add_argument("-k", "--api_key", type=str, required=True, help="Moralis API key for data fetching")
    parser.add_argument(
        "-t",
        "--timeframe",
        type=str,
        choices=["1s", "10s", "30s", "1min", "5min", "10min", "30min", "1h", "4h", "12h", "1d", "1w", "1M"],
        default="1h",
        help="Candlestick period (timeframe)"
    )
    parser.add_argument(
        "-c",
        "--currency",
        type=str,
        choices=["usd", "native"],
        default="usd",
        help="Currency paired with token to fetch"
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default="./output.csv",
        help="Location to save the output CSV file"
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        required=False,
        help="File index to append to the end of data batch filename. Usually set to dataset files count inside a folder. Defaults to 0."
    )
    args = parser.parse_args()
    if False in util.is_valid_iso_format([args.start] + [] if args.end is None else [args.end]):
        print("Incorrect date format")
        return
    if not util.is_valid_location(args.location):
        print("Incorrect output file location")
        return
    try:
        location: str = args.location
        location = location if location.endswith(".csv") else location + ".csv"
        file_index = max(0, args.index)

        def process_file(accumulated: list[Candle]):
            nonlocal file_index
            if len(accumulated) < 200: return
            write(accumulated, location.replace(".csv", f"_{file_index}.csv"))
            file_index += 1

        def on_completed(accumulated: list[Candle]):
            process_file(accumulated)

        def on_error(_: Exception, accumulated: list[Candle]):
            process_file(accumulated)

        stream_fetch(
            args.addr,
            args.start,
            on_completed,
            args.end,
            args.api_key,
            args.timeframe,
            args.currency,
            on_error
        )
    except Exception as e:
        print(f"Failed to fetch candlestick data for pair {args.addr}, error {e}")


if __name__ == "__main__":
    main()
