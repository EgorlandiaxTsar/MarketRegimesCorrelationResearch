import argparse
import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import optuna
import pandas as pd

import src.utils as util
from src.env.simulation import Agent, Market, MarketException
from src.env.strategy import DipsStrategy
from src.env.data import ParametersSet

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)


def score(input: pd.DataFrame, target: pd.DataFrame, params: ParametersSet) -> float:
    scores: list[float] = []
    market = Market(target.copy())
    agent = Agent(capital=1000, commission=0.01)
    strategy = DipsStrategy(
        market,
        agent,
        detection_period=params.detection_period,
        drop_threshold=params.drop_threshold,
        reversal_threshold=params.reversal_threshold,
        tp_pct=params.tp_pct,
        sl_pct=params.sl_pct
    )
    strategy.evaluate(verbose=False)
    scores.append(
        agent.total_capital + len(agent.actions) // 2 if agent.total_capital > 20 else agent.total_capital
    )
    return min(scores)


def objective(trial: optuna.Trial, input: pd.DataFrame, target: pd.DataFrame) -> ParametersSet:
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
    try: params.result = score(input, target, params)
    except MarketException: pass
    return params


def hybrid_optimize(name: str, input: pd.DataFrame, target: pd.DataFrame, tpe_trials: int, cma_trials: int) -> tuple[float, list[ParametersSet]]:
    trials: list[ParametersSet] = []
    best, best_index = 0, 0
    tpe_study = optuna.create_study(
        storage=None, # sqlite:///data/db/optuna.db, load_if_exists
        direction="maximize",
        study_name=f"{name}_TPE",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10)
    )
    for i in range(tpe_trials):
        if i - best_index >= int(tpe_trials * 0.2):
            break
        trial = tpe_study.ask()
        result = objective(trial, input, target)
        trials.append(result)
        tpe_study.tell(trial, result.result)
        if result.result > best:
            best, best_index = result.result, i
    best_index = 0
    best_trial = tpe_study.best_trial.params
    cma_study = optuna.create_study(
        storage=None, # sqlite:///data/db/optuna.db, load_if_exists
        direction="maximize",
        study_name=f"{name}_CmaEs",
        sampler=optuna.samplers.CmaEsSampler(x0={
            "detection_period": best_trial["detection_period"],
            "drop_threshold": best_trial["drop_threshold"],
            "reversal_threshold": best_trial["reversal_threshold"],
            "tp_pct": best_trial["tp_pct"],
            "sl_pct": best_trial["sl_pct"]
        }, sigma0=0.05, popsize=20)
    )
    for i in range(cma_trials):
        if i - best_index >= int(tpe_trials * 0.2):
            break
        trial = cma_study.ask()
        result = objective(trial, input, target)
        trials.append(result)
        cma_study.tell(trial, result.result)
        if result.result > best:
            best, best_index = result.result, i
    best_study = tpe_study if tpe_study.best_value > cma_study.best_value else cma_study
    return best_study.best_value, sorted(trials, key=lambda t: t.result if t.result is not None else float("-inf"), reverse=True)[:20]


def worker(group: list[tuple[str, str, str]], n_trials: int, jobs: int, location: str):
    for e in group:
        prev, curr, feature = e
        df_prev, df = pd.read_csv(prev), pd.read_csv(curr)
        value, trials = hybrid_optimize(feature, df_prev, df, math.ceil(n_trials * 1), 1)
        print(f"{prev:32} | {value:24} | {str(trials[0].detection_period):2} | {str(trials[0].drop_threshold):24} | {str(trials[0].reversal_threshold):24} | {str(trials[0].tp_pct):24} | {str(trials[0].sl_pct):24}")
        with open(location + "/" + Path(feature).name.replace("features", "params"), "w") as f:
            json.dump(
                {
                    "metadata": {"feature": feature},
                    "best": [{
                        "value": e.result,
                        "params": {
                            "detection_period": e.detection_period,
                            "drop_threshold": e.drop_threshold,
                            "reversal_threshold": e.reversal_threshold,
                            "tp_pct": e.tp_pct,
                            "sl_pct": e.sl_pct
                        }
                    } for e in trials]
                }, f
            )


def main():
    parser = argparse.ArgumentParser(
        prog="parameters_finder.py",
        description="Features datasets evaluator utility script"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Directory from where to extract all JSON features (nested directories also)"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory where to store parameters output")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=0,
        help="Computation threads count. Default 0 to use one thread for each dataset"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        required=False,
        default=0,
        help="Optuna computation subthreads count. Default 0 to use all available CPU cores"
    )
    parser.add_argument("-i", "--iterations", type=int, required=True, help="Optuna trials count")
    args = parser.parse_args()
    if not util.is_valid_location(args.data) or not util.is_valid_location(args.output):
        print("Incorrect input/output file locations")
        return
    jobs: int = os.cpu_count() if args.jobs == 0 else max(1, args.jobs)
    args.iterations = max(1, args.iterations)
    features = [str(p) for p in Path(args.data).rglob("*.json")]
    datasets: list[tuple[str, str, str]] = []
    for feature in features:
        with open(feature, "r") as f: info = json.load(f)
        datasets.append((info["metadata"]["input"], info["metadata"]["target"], feature))
    if len(datasets) == 0: return
    datasets_per_thread = math.ceil(len(datasets) / args.threads if args.threads > 0 else 1)
    groups = []
    buf, index = [], 0
    for dataset in datasets:
        if index >= datasets_per_thread:
            index = 0
            groups.append(buf.copy())
            buf = []
        buf.append(dataset)
        index += 1
    if len(buf) != 0: groups.append(buf)
    buf = []
    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(worker, group, args.iterations, jobs, args.output) for group in groups]
        print("Parameters search started")
        for e in futures: e.result()
        print("Parameters search completed")


if __name__ == "__main__":
    main()
