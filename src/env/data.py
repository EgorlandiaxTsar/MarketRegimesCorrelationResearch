import json
import os

import numpy as np
import pandas as pd


class Candle:
    def __init__(self, open, high, low, close, volume, timestamp):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timestamp = timestamp


class Dataset:
    def __init__(self, name: str, location: str, lazy: bool = True, regimes_grid: pd.DataFrame | None = None):
        self.name = name
        self.location = location
        self.input: list[Candle] | None = None
        self.target: list[Candle] | None = None
        self.strategy_params: list[ParametersSet] | None = None
        self.features: Features | None = None
        self.regime: int | None = None
        self.regimes_grid: pd.DataFrame | None = regimes_grid
        if not lazy: self.load()

    def load(self):
        self.load_prices()
        self.load_features()
        self.load_params()
        self.load_regime()

    def load_prices(self):
        self.target = self._load_data(self.name)

    def load_features(self):
        path = self._find_file(f"features_{self.name}")
        if path is None: return
        with open(path, "r") as f:
            data = json.load(f)
        input_file = data.get("metadata", {}).get("input")
        if input_file: self.input = self._load_data(input_file.split("data_")[-1].replace(".csv", ""))
        s = data["features"]["scalar"]
        g = data["features"]["graphical"]
        scalar = ScalarFeatures(
            std=s["std"],
            rstd=s["rstd"],
            ma_distance_ratio=s["ma_distance_ratio"],
            rsi_crossovers=s["rsi_crosses"][0],
            rsi_crossunders=s["rsi_crosses"][1],
            entropy=s["entropy"],
            regression_slope=s["regression_slope"],
            regression_sqr=s["regression_sqr"],
            volatility_clustering=s["volatility_clustering"],
            max_drawdown_ratio=s["max_drawdown_ratio"],
            time_to_trough_ratio=s["time_to_trough_ratio"],
            recovery_ratio=s["recovery_ratio"],
            path_length_ratio=s["path_length_ratio"],
            sign_change_ratio=s["sign_change_ratio"],
            drawdown_convexity=s["drawdown_convexity"],
            drop_velocity=s["drop_velocity"]
        )
        graphical = GraphicalFeatures(
            ma15=g["ma15"],
            rsi15=g["rsi15"],
            atr60=g["atr60"]
        )
        self.features = Features(
            scalar=scalar,
            graphical=graphical
        )

    def load_params(self):
        path = self._find_file(f"params_{self.name}")
        if path is None: return
        with open(path, "r") as f:
            data = json.load(f)
        self.strategy_params = []
        for e in data.get("best", []):
            self.strategy_params.append(
                ParametersSet(
                    result=e["value"],
                    detection_period=e["params"]["detection_period"],
                    drop_threshold=e["params"]["drop_threshold"],
                    reversal_threshold=e["params"]["reversal_threshold"],
                    tp_pct=e["params"]["tp_pct"],
                    sl_pct=e["params"]["sl_pct"],
                )
            )

    def load_regime(self):
        df = self.regimes_grid if self.regimes_grid is not None else pd.read_csv(f"{self.location}/dataset_regime_map.csv")
        if self.name in df.values:
            self.regime = df.loc[df['dataset'] == self.name]["regime"].iloc[0]

    def _find_file(self, prefix: str, extensions: tuple = (".csv", ".json")) -> str | None:
        for root, _, files in os.walk(self.location):
            for file in files:
                if file.startswith(prefix) and file.endswith(extensions):
                    return os.path.join(root, file)
        return None

    def _load_data(self, name: str) -> list[Candle] | None:
        path = self._find_file(f"data_{name}")
        if path is None: return None
        df = pd.read_csv(path)
        return [Candle(row["o"], row["h"], row["l"], row["c"], row["v"], row["t"]) for _, row in df.iterrows()]


class ParametersSet:
    def __init__(
            self,
            result: float,
            detection_period: int,
            drop_threshold: float,
            reversal_threshold: float,
            tp_pct: float,
            sl_pct: float
    ):
        self.result = np.float64(result)
        self.detection_period = detection_period
        self.drop_threshold = np.float64(drop_threshold)
        self.reversal_threshold = np.float64(reversal_threshold)
        self.tp_pct = np.float64(tp_pct)
        self.sl_pct = np.float64(sl_pct)


class ScalarFeatures:
    def __init__(
            self,
            std: float,
            rstd: float,
            ma_distance_ratio: float,
            rsi_crossovers: int,
            rsi_crossunders: int,
            entropy: float,
            regression_slope: float,
            regression_sqr: float,
            volatility_clustering: float,
            max_drawdown_ratio: float,
            time_to_trough_ratio: float,
            recovery_ratio: float,
            path_length_ratio: float,
            sign_change_ratio: float,
            drawdown_convexity: float,
            drop_velocity: float

    ):
        self.std = np.float64(std)
        self.rstd = np.float64(rstd)
        self.ma_distance_ratio = np.float64(ma_distance_ratio)
        self.rsi_crossovers = rsi_crossovers
        self.rsi_crossunders = rsi_crossunders
        self.entropy = np.float64(entropy)
        self.regression_slope = np.float64(regression_slope)
        self.regression_sqr = np.float64(regression_sqr)
        self.volatility_clustering = np.float64(volatility_clustering)
        self.max_drawdown_ratio = np.float64(max_drawdown_ratio)
        self.time_to_trough_ratio = np.float64(time_to_trough_ratio)
        self.recovery_ratio = np.float64(recovery_ratio)
        self.path_length_ratio = np.float64(path_length_ratio)
        self.sign_change_ratio = np.float64(sign_change_ratio)
        self.drawdown_convexity = np.float64(drawdown_convexity)
        self.drop_velocity = np.float64(drop_velocity)


class GraphicalFeatures:
    def __init__(self, ma15: list[float], rsi15: list[float], atr60: list[float]):
        self.ma15 = np.array(ma15, dtype=np.float64)
        self.rsi15 = np.array(rsi15, dtype=np.float64)
        self.atr60 = np.array(atr60, dtype=np.float64)


class Features:
    def __init__(self, scalar: ScalarFeatures, graphical: GraphicalFeatures):
        self.scalar = scalar
        self.graphical = graphical
