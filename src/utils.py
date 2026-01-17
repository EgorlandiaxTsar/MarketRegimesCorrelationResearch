import os
import shutil
import math
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.env.data import Dataset, ScalarFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

LEGEND_POSITION = "upper left"
ZERO_SHIFT = 1e-12


def rsi(df: pd.DataFrame, length: int = 14) -> np.ndarray[np.float64]:
    delta = df["c"].diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gains.ewm(alpha=1 / length, adjust=False).mean(), losses.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()
    rs = avg_gain / (avg_loss + ZERO_SHIFT)
    return (100 - (100 / (1 + rs))).to_numpy(dtype=np.float64)


def ma(df: pd.DataFrame, length: int = 15) -> np.ndarray[np.float64]:
    return df["c"].rolling(length).mean().to_numpy(dtype=np.float64)


def atr(df: pd.DataFrame, length: int = 14) -> np.ndarray[np.float64]:
    df = df.copy()
    high = df["h"]
    low = df["l"]
    close = df["c"]
    df["tr0"], df["tr1"], df["tr2"] = abs(high - low), abs(high - close.shift()), abs(low - close.shift())
    tr = df[["tr0", "tr1", "tr2"]].max(axis=1)
    atr = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return atr.to_numpy(dtype=np.float64)


def std(df: pd.DataFrame, target: str = "c") -> np.float64:
    return np.sqrt(df[target].to_numpy(dtype=np.float64).var())


def rstd(df: pd.DataFrame, target: str = "c") -> np.float64:
    values = df[target].to_numpy(dtype=np.float64)
    mean = np.mean(values)
    if mean == 0: return 0.0
    return np.std(values) / mean


def ma_distance_ratio(df: pd.DataFrame, target: str = "c", ma: str = "ma15") -> np.float64:
    return np.mean(abs((df[target] - df[ma]) / (df[ma] + ZERO_SHIFT)))


def candle_spread_pct(df: pd.DataFrame) -> np.ndarray[np.float64]:
    return ((df["h"] - df["l"]) / (df["c"] + ZERO_SHIFT)).to_numpy(dtype=np.float64)


def candle_wick_spread_pct(df: pd.DataFrame) -> np.ndarray[np.float64]:
    return (((df["h"] - np.maximum(df["o"], df["c"])) + (np.minimum(df["o"], df["c"]) - df["l"])) / (
                abs(df["c"] - df["o"]) + ZERO_SHIFT)).to_numpy(dtype=np.float64)


def rsi_crosses_count(
        df: pd.DataFrame,
        target: str = "rsi14",
        min_lim: np.float64 = 30.0,
        max_lim: np.float64 = 70.0
) -> tuple[int, int]:
    rsi = df[target].to_numpy(dtype=np.float64)
    previous, current = rsi[:-1], rsi[1:]
    crossovers = np.sum((previous < min_lim) & (current >= min_lim))
    crossunders = np.sum((previous > max_lim) & (current <= max_lim))
    return int(crossovers), int(crossunders)


def shannon_entropy(df: pd.DataFrame, target: str = "c", bins: int = 20) -> np.float64:
    returns = np.log(df[target] / (df[target].shift(1) + ZERO_SHIFT)).dropna().to_numpy(dtype=np.float64)
    hist, _ = np.histogram(returns, bins=bins, density=False)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return np.float64(entropy)


def regression_slope(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    x = np.arange(len(prices), dtype=np.float64)
    slope, _ = np.polyfit(x, prices, 1)
    normalized_slope = slope / (np.mean(prices) + ZERO_SHIFT)
    return np.float64(normalized_slope)


def regression_r2(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    n = len(prices)
    if n < 2: return np.float64(0.0)
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, prices, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((prices - predicted) ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2) + ZERO_SHIFT
    r2 = 1 - (ss_res / ss_tot)
    return np.float64(r2)


def volatility_clustering(df: pd.DataFrame, target: str = "c", lag: int = 1) -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    rets = np.abs(np.diff(np.log(prices + 1e-12)))
    if len(rets) <= lag: return np.float64(0.0)
    v1, v2 = rets[:-lag], rets[lag:]
    corr = np.corrcoef(v1, v2)[0, 1]
    if np.isnan(corr): return np.float64(0.0)
    return np.float64(corr)


def max_drawdown_ratio(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    peak = prices[0]
    max_dd = 0.0
    for p in prices[1:]:
        if p > peak:
            peak = p
        else:
            dd = (peak - p) / (peak + ZERO_SHIFT)
            if dd > max_dd:
                max_dd = dd
    return np.float64(max_dd)


def time_to_trough_ratio(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    peak_idx = 0
    peak_price = prices[0]
    trough_idx = 0
    trough_price = prices[0]
    for i in range(1, len(prices)):
        if prices[i] > peak_price:
            peak_price = prices[i]
            peak_idx = i
            trough_price = prices[i]
            trough_idx = i
        elif prices[i] < trough_price:
            trough_price = prices[i]
            trough_idx = i
    if trough_idx <= peak_idx:
        return np.float64(0.0)
    return np.float64((trough_idx - peak_idx) / len(prices))


def recovery_ratio(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    peak = prices[0]
    trough = prices[0]
    for i in range(1, len(prices)):
        if prices[i] > peak:
            peak = prices[i]
            trough = prices[i]
        elif prices[i] < trough:
            trough = prices[i]
    max_dd = (peak - trough) / (peak + ZERO_SHIFT)
    if max_dd == 0:
        return np.float64(0.0)
    recovery = (prices[-1] - trough) / (peak - trough + ZERO_SHIFT)
    return np.float64(recovery)


def path_length_ratio(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    log_prices = np.log(prices + ZERO_SHIFT)
    returns = np.diff(log_prices)
    path_length = np.sum(np.abs(returns))
    net_displacement = abs(log_prices[-1] - log_prices[0])
    if net_displacement == 0:
        return np.float64(0.0)
    return np.float64(path_length / net_displacement)


def sign_change_ratio(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    returns = np.diff(prices)
    if len(returns) < 2:
        return np.float64(0.0)
    signs = np.sign(returns)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    return np.float64(sign_changes / len(returns))


def drawdown_convexity(df: pd.DataFrame, target: str = "c") -> np.float64:
    prices = df[target].to_numpy(dtype=np.float64)
    peak = prices[0]
    drawdowns = []
    for p in prices:
        if p > peak:
            peak = p
        drawdowns.append((peak - p) / (peak + ZERO_SHIFT))
    drawdowns = np.array(drawdowns, dtype=np.float64)
    x = np.arange(len(drawdowns), dtype=np.float64)
    if len(drawdowns) < 3:
        return np.float64(0.0)
    try:
        a, _, _ = np.polyfit(x, drawdowns, 2)
    except np.linalg.LinAlgError:
        return np.float64(0.0)
    return np.float64(a)


def drop_velocity(df: pd.DataFrame, target: str = "c") -> np.float64:
    dd = max_drawdown_ratio(df, target)
    ttr = time_to_trough_ratio(df, target)
    if ttr == 0:
        return np.float64(0.0)
    return np.float64(dd / ttr)


def kmeans_inertia(x: np.ndarray, start: int = 1, end: int = 100):
    inertia: list[float] = []
    for k in range(start, end):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x)
        inertia.append(kmeans.inertia_)
    plot(inertia, "Inertia", "Inertia")


def kmeans_silhouette(x: np.ndarray, start: int = 2, end: int = 100) -> tuple[int, np.float64, pd.DataFrame]:
    silhouette_scores = []
    best_k, max_score = 0, 0
    for k in range(start, end):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(x)
        score = silhouette_score(x, labels, random_state=42)
        silhouette_scores.append({"k": k, "score": score})
        max_score = max(score, max_score)
        best_k = k if max_score == score else best_k
    return best_k, np.float64(max_score), pd.DataFrame(silhouette_scores)


def select_cluster_groups(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray, count: int) -> np.ndarray[np.ndarray[np.float64 | int]]:
    indices = []
    for k in range(len(centroids)):
        cluster_indexes = np.where(labels == k)[0]
        if len(cluster_indexes) < count: continue
        distances = np.linalg.norm(x[cluster_indexes] - centroids[k], axis=1)
        indices.extend(cluster_indexes[np.argsort(distances)][:count])
    return np.array(indices)


def price_plot(
        candles: pd.DataFrame,
        indicators: list[tuple[str, str, float]] = [],
        separated_indicators: list[tuple[str, str, float]] = [],
        plot_size: tuple[int, int] = (20, 6),
        main_color: str = "b"
):
    labels = ["Close Price"]
    plt.figure(figsize=plot_size)
    plt.plot(candles["c"].values, color=main_color)
    for indicator in indicators:
        plt.plot(indicator[2].values, color=indicator[0])
        labels.append(indicator[1])
    plt.legend(labels, loc=LEGEND_POSITION)
    plt.grid(True)
    plt.show()
    for indicator in separated_indicators:
        label = [indicator[1]]
        val_min = indicator[2]
        val_max = indicator[3]
        plt.figure(figsize=plot_size)
        if val_min >= 0:
            plt.plot(val_min, color="w")
            label.insert(0, "Min")
        if val_max > val_min:
            plt.plot(val_max, color="w")
            label.insert(1, "Max")
        plt.plot(indicator[4].values, color=indicator[0])
        plt.legend(label, loc=LEGEND_POSITION)
        plt.grid(True)
        plt.show()


def plot(data: list[float | int], title: str, legend: str):
    plt.figure(figsize=(20, 6))
    plt.ticklabel_format(axis="y", style="plain")
    plt.plot(data, color="k", label=legend)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def clip(value: int | float, min_value: int | float, max_value: int | float) -> int | float:
    return max(min_value, min(value, max_value))


def is_valid_location(location: str) -> bool:
    folder = os.path.dirname(location)
    if folder and not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception:
            return False
    return True


def is_valid_iso_format(values: str | list[str]) -> bool | list[bool]:
    def _validate_iso(date_str: str) -> bool:
        try:
            datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    if isinstance(values, str):
        return _validate_iso(values)
    return [_validate_iso(v) for v in values]


def __loader(group: list[Dataset]) -> list[Dataset]:
    for e in group: e.load()
    return group


def load_datasets(path: str, shift: int, max: int = -1, regimes_grid: pd.DataFrame = None) -> list[Dataset]:
    dataset_files: list[str] = []
    shift_ends = tuple([f"_{i}.csv" for i in range(0, shift)])
    for e in Path(f"{path}/sliced").rglob("*.csv"):
        name = e.name.replace(".csv", "").replace("data_", "")
        if not e.name.endswith(shift_ends): dataset_files.append(name)
    datasets: list[Dataset] = [Dataset(e, path, True, regimes_grid) for e in dataset_files][:max]
    datasets_per_thread = math.ceil(len(datasets) / os.cpu_count())
    groups, buf, index = [], [], 0
    for dataset in datasets:
        if index >= datasets_per_thread:
            groups.append(buf)
            buf, index = [], 0
        buf.append(dataset)
        index += 1
    if len(buf) > 0: groups.append(buf)
    buf = []
    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(__loader, group) for group in groups]
        datasets = []
        for e in futures: datasets.extend(e.result())
    return datasets


def dataset_to_dataframe(datasets: list[Dataset]) -> pd.DataFrame:
    def features_row(dataset: Dataset) -> dict[str, any]:
        return {
            "dataset": dataset.name,
            "std": scalar.std,
            "rstd": scalar.rstd,
            "ma_distance_ratio": scalar.ma_distance_ratio,
            "rsi_crossovers": scalar.rsi_crossovers,
            "rsi_crossunders": scalar.rsi_crossunders,
            "entropy": scalar.entropy,
            "regression_slope": scalar.regression_slope,
            "regression_sqr": scalar.regression_sqr,
            "volatility_clustering": scalar.volatility_clustering,
            "max_drawdown_ratio": scalar.max_drawdown_ratio,
            "time_to_trough_ratio": scalar.time_to_trough_ratio,
            "recovery_ratio": scalar.recovery_ratio,
            "path_length_ratio": scalar.path_length_ratio,
            "sign_change_ratio": scalar.sign_change_ratio,
            "drawdown_convexity": scalar.drawdown_convexity,
            "drop_velocity": scalar.drop_velocity
        }
    rows = []
    for dataset in datasets:
        if dataset.features is None: continue
        scalar: ScalarFeatures = dataset.features.scalar
        row = features_row(dataset)
        if dataset.regime is not None: row["regime"] = dataset.regime
        if dataset.strategy_params is None:
            rows.append(row)
            continue
        for trial, params in enumerate(dataset.strategy_params):
            row_copy = row.copy()
            if params.result is None: continue
            row_copy["trial_rank"] = trial
            row_copy["profit"] = params.result
            row_copy["detection_period"] = params.detection_period
            row_copy["drop_threshold"] = params.drop_threshold
            row_copy["reversal_threshold"] = params.reversal_threshold
            row_copy["tp_pct"] = params.tp_pct
            row_copy["sl_pct"] = params.sl_pct
            rows.append(row_copy)
    return pd.DataFrame(rows)


def copy_files(files: list[str], location: str):
    dst = Path(location)
    dst.mkdir(parents=True, exist_ok=True)
    for f in files: shutil.copy2(f, dst) 
