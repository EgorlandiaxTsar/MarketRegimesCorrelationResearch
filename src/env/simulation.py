from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.env.data import Candle


class MarketException(Exception):
    pass


class MarketAction:
    def __init__(self, candle: Candle, amount: float, type: Literal["buy", "sell"]):
        self.candle = candle
        self.amount = amount
        self.type = type


class Market:
    def __init__(self, data: pd.DataFrame):
        self.data: list[Candle] = [Candle(row["t"], row["o"], row["h"], row["l"], row["c"], row["v"]) for _, row in
                                   data.iterrows()]
        self.current_index: int = 0

    def forward(self) -> Candle | None:
        if self.current_index < len(self.data):
            candle = self.data[self.current_index]
            self.current_index += 1
            return candle
        else:
            return None

    def backward(self) -> Candle:
        self.current_index = max(0, self.current_index - 1)
        return self.data[self.current_index]

    def reset(self):
        self.current_index = 0

    def history(self) -> list[Candle]:
        end_index = max(0, self.current_index)
        return self.data[0:end_index]


class Agent:
    def __init__(self, capital: float, commission: float):
        self.asset_capital, self.capital, self.total_capital, self.initial_capital = 0, capital, capital, capital
        self.commission = commission
        self.actions: list[MarketAction] = []
        self.capital_change: list[float] = []

    @property
    def pnl(self) -> float:
        return self.total_capital - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        return (self.total_capital / self.initial_capital) - 1.0

    @property
    def returns(self) -> np.ndarray:
        equity = np.array(self.capital_change, dtype=np.float64)
        if len(equity) < 2:
            return np.array([], dtype=np.float64)
        return np.diff(equity) / equity[:-1]

    @property
    def max_drawdown(self) -> float:
        equity = np.array(self.capital_change, dtype=np.float64)
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / (peak + 1e-12)
        return float(np.max(drawdowns))

    @property
    def trade_count(self) -> int:
        buys = sum(1 for a in self.actions if a.type == "buy")
        sells = sum(1 for a in self.actions if a.type == "sell")
        return min(buys, sells)
    

    def buy(self, candle: Candle, amount: float):
        price = (candle.close * amount) * (1 + self.commission)
        if price > self.capital:
            raise MarketException("Not enough capital to buy")
        self.asset_capital += amount
        self.capital -= price
        self.total_capital = self.capital + self.asset_capital * candle.close
        self.actions.append(MarketAction(candle, amount, "buy"))
        self.capital_change.append(self.total_capital)

    def sell(self, candle: Candle, amount: float):
        if amount > self.asset_capital:
            raise MarketException("Not enough assets to sell")
        price = (candle.close * amount) * (1 - self.commission)
        self.asset_capital -= amount
        self.capital += price
        self.total_capital = self.capital + self.asset_capital * candle.close
        self.actions.append(MarketAction(candle, amount, "sell"))
        self.capital_change.append(self.total_capital)

    def hold(self, candle: Candle):
        self.total_capital = self.capital + self.asset_capital * candle.close
        self.capital_change.append(self.total_capital)

    def plot_trades(self, market: Market):
        buys = [action for action in self.actions if action.type == "buy"]
        sells = [action for action in self.actions if action.type == "sell"]
        price = market.data
        closes = [candle.close for candle in price]
        buy_x = [price.index(buy.candle) for buy in buys]
        buy_y = [buy.candle.close for buy in buys]
        sell_x = [price.index(sell.candle) for sell in sells]
        sell_y = [sell.candle.close for sell in sells]
        plt.figure(figsize=(20, 6))
        plt.plot(closes, color="k", label="Price")
        plt.scatter(buy_x, buy_y, color="green", marker="^", label="Buys", s=60, alpha=1)
        plt.scatter(sell_x, sell_y, color="red", marker="v", label="Sells", s=60, alpha=1)
        plt.title("Market Trades")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_capital_change(self):
        plt.figure(figsize=(20, 6))
        plt.plot(self.capital_change, label="Total Capital", color="k")
        plt.title("Total Capital Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
