import time
import threading
import os
import math
import warnings
import argparse
import optuna
import pandas as pd
import numpy as np
import src.utils as util

from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
from src.env.data import ParametersSet
from src.env.simulation import Agent, Market, MarketException
from src.env.strategy import DipsStrategy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.env.data import ScalarFeatures


SCALER = StandardScaler()
TARGET_FEATURES: list[str] = ["rstd", "rsi_crossovers", "regression_slope", "regression_sqr"]
COLUMNS = [
    "regime",
    "current_trial",
    "best_value",
    "detection_period",
    "drop_threshold",
    "reversal_threshold",
    "tp_pct",
    "sl_pct",
]

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)


def monitor_table(shared_table):
    while True:
        time.sleep(5)
        if not shared_table:
            continue
        snapshot = dict(shared_table)
        df = pd.DataFrame.from_dict(snapshot, orient="index")[COLUMNS]
        os.system("cls" if os.name == "nt" else "clear")
        print(df.sort_values("regime").to_string(index=False))


def score(group: pd.DataFrame, params: ParametersSet, location: str) -> float:
    scores = []
    for _, e in group.iterrows():
        market = Market(pd.read_csv(f"{location}/sliced/data_{e['dataset']}.csv"))
        agent = Agent(capital=1000, commission=0.01)
        strategy = DipsStrategy(
            market, agent,
            params.detection_period,
            params.drop_threshold,
            params.reversal_threshold,
            params.tp_pct,
            params.sl_pct
        )
        strategy.evaluate(verbose=False)
        final_cap = agent.total_capital
        trades = len(agent.actions)
        max_dd = agent.max_drawdown
        activity = np.tanh(trades / 10)
        cap_score = final_cap / 1000
        dd_penalty = max_dd * 2.0
        scores.append(cap_score + activity - dd_penalty)
    scores = np.array(scores)
    return scores.mean() - 0.5 * scores.std()


def objective(trial: optuna.Trial, group: pd.DataFrame, location: str) -> ParametersSet:
    detection_period = trial.suggest_int("detection_period", 1, 6)
    tp_pct = trial.suggest_float("tp_pct", 0.01, 0.1, step=0.0001)
    sl_pct = trial.suggest_float("sl_pct", tp_pct * 0.1, tp_pct * 0.9, step=0.0001)
    drop_threshold = trial.suggest_float("drop_threshold", 0.03, 0.15, step=0.0001)
    reversal_threshold = trial.suggest_float(
        "reversal_threshold",
        drop_threshold * 0.3,
        drop_threshold * 0.9,
        step=0.0001,
    )
    params = ParametersSet(-10000, detection_period, drop_threshold, reversal_threshold, tp_pct, sl_pct)
    try: params.result = score(group, params, location)
    except MarketException: pass
    return params


def optimize(name: str, n_trials: int, group: pd.DataFrame, location: str, shared_table: dict[str, any]) -> tuple[float, list[ParametersSet]]:
    trials: list[ParametersSet] = []
    best: dict[str, any] = {}
    best_index = 0
    study = optuna.create_study(
        storage=None,
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=100
        )
    )
    for i in range(n_trials):
        if i - best_index >= 200 or len(group) == 0:
            break
        trial = study.ask()
        result = objective(trial, group, location)
        trials.append(result)
        study.tell(trial, result.result)            
        current_best = best.get("value", float("-inf"))
        stats = shared_table[name].copy()
        stats["current_trial"] = i
        if result.result is not None and result.result > current_best:
            best_index = i
            best["value"] = result.result
            stats["best_value"] = result.result
            stats["detection_period"] = result.detection_period
            stats["drop_threshold"] = result.drop_threshold
            stats["reversal_threshold"] = result.reversal_threshold
            stats["tp_pct"] = result.tp_pct
            stats["sl_pct"] = result.sl_pct
        shared_table[name] = stats
    return study.best_value, sorted(trials, key=lambda t: t.result if t.result is not None else float("-inf"), reverse=True)[:20]


def worker(groups: list[pd.DataFrame], n_trials: int, location: str, shared_table: dict[str, any]) -> list[tuple[int, float, list[ParametersSet]]]:
    output = []
    for group in groups:
        name = str(group["regime"].iloc[0])
        value, trials = optimize(name, n_trials, group, location, shared_table)
        output.append((int(group["regime"].iloc[0]), value, trials))
    return output


def compute(df: pd.DataFrame, k: int, n_trials: int, location: str) -> pd.DataFrame:
    manager = Manager()
    shared_table = manager.dict()
    for i in range(k):
        shared_table[str(i)] = {"regime": i, "current_trial": 0, "best_value": 0, "detection_period": 0, "drop_threshold": 0, "reversal_threshold": 0, "tp_pct": 0, "sl_pct": 0}
    monitor = threading.Thread(target=monitor_table, args=(shared_table,), daemon=True)
    monitor.start()
    regime_dfs: list[pd.DataFrame] = []
    for regime in range(k): regime_dfs.append(df[df["regime"] == regime])
    regimes_per_thread = math.ceil(len(regime_dfs) / os.cpu_count())
    groups, buf, index = [], [], 0
    for regime_df in regime_dfs:
        if index >= regimes_per_thread:
            groups.append(buf)
            buf, index = [], 0
        buf.append(regime_df)
        index += 1
    if len(buf) > 0: groups.append(buf)
    buf = []
    results = {
        "regime": [],
        "value": [],
        "detection_period": [],
        "drop_threshold": [],
        "reversal_threshold": [],
        "tp_pct": [],
        "sl_pct": []
    }
    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(worker, group, n_trials, location, shared_table) for group in groups]
        for future in futures:
            for e in future.result():
                regime, _, trials = e
                for trial in trials:
                    results["regime"].append(regime)
                    results["value"].append(trial.result)
                    results["detection_period"].append(trial.detection_period)
                    results["drop_threshold"].append(trial.drop_threshold)
                    results["reversal_threshold"].append(trial.reversal_threshold)
                    results["tp_pct"].append(trial.tp_pct)
                    results["sl_pct"].append(trial.sl_pct)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        prog="regimes_parameters_finder.py",
        description="Optimal parameters finder for regimes-grouped datasets utility script"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Root directory of target dataset"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="File where to output results CSV file"
    )
    parser.add_argument(
        "-f",
        "--shift",
        type=int,
        default=1,
        help="Dataset shifting. Defaults to 0"
    )
    parser.add_argument(
        "-k",
        "--classes",
        type=int,
        required=True,
        help="Classes count"
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        required=True,
        help="Trials count for each study"
    )
    args = parser.parse_args()
    if not util.is_valid_location(args.data) or not util.is_valid_location(args.output):
        print("Incorrect input/output locations")
        return
    shift, classes, trials = max(1, args.shift), max(2, args.classes), max(10, args.trials)
    print("Loading datasets")
    datasets = []
    for e in util.load_datasets(args.data, shift):
        if e.regime is not None:
            datasets.append(e)
    print("Datasets loaded")
    df = util.dataset_to_dataframe(datasets)
    df = df[df["trial_rank"] == 0]
    regime_counts = pd.Series(df["regime"]).value_counts()
    valid_regimes = regime_counts[regime_counts >= 10].index.values
    df = df[df["regime"].isin(valid_regimes)]
    classes = len(valid_regimes)
    print(f"Loaded {len(df)} items, valid classes {classes}")
    results = compute(df, classes, trials, args.data)
    results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
