from abc import ABC, abstractmethod

import src.utils as util
from src.env.data import Candle
from src.env.simulation import Market, MarketException, Agent


class Strategy(ABC):
    def __init__(self, market: Market, agent: Agent):
        self.market = market
        self.agent = agent

    @abstractmethod
    def forward(self) -> bool:
        pass

    def evaluate(self, verbose: bool = True):
        alive = True
        while alive:
            try:
                alive = self.forward()
            except MarketException as e:
                print(f"Market Exception: {e}")
                break
        if verbose:
            self.agent.plot_trades(self.market)
            self.agent.plot_capital_change()

    def _skip(self) -> Candle:
        candle = self.market.forward()
        self.agent.hold(candle)
        return candle


class DipsStrategy(Strategy):
    def __init__(
            self,
            market: Market,
            agent: Agent,
            detection_period: float = 3,
            drop_threshold: float = 0.07,
            reversal_threshold: float = 0.035,
            tp_pct: float = 0.05,
            sl_pct: float = 0.015
    ):
        super().__init__(market, agent)
        self.tp_pct, self.sl_pct = tp_pct, sl_pct
        self.tp_target, self.sl_target = None, None
        self.detection_period, self.drop_threshold, self.reversal_threshold = detection_period, drop_threshold, reversal_threshold
        self.is_position_opened, self.drop_price, self.period_start_price, self.periods = False, None, None, None

    def forward(self) -> bool:
        candle = self.market.forward()
        if candle is None: return False
        if self.periods is None or self.periods > self.detection_period: self.__reset_period(candle)
        action_executed = False
        price_change = (candle.close - self.period_start_price) / self.period_start_price
        is_drop, is_reversal = price_change <= -self.drop_threshold, False if self.drop_price is None else (candle.close - self.drop_price) / self.drop_price >= self.reversal_threshold
        if (self.drop_price is not None and candle.close < self.drop_price) or (
                is_drop and self.drop_price is None): self.drop_price = candle.close
        if is_reversal and not self.is_position_opened:
            buy_amount = (self.agent.capital / candle.close) * (1 - self.agent.commission)
            self.agent.buy(candle, buy_amount)
            self.is_position_opened = True
            self.tp_target, self.sl_target = self.__calculate_exit_targets(candle)
            action_executed = True
        elif self.is_position_opened:
            if candle.close >= self.tp_target:
                self.tp_target, self.sl_target = self.__calculate_exit_targets(candle)
            elif candle.close <= self.sl_target:
                self.agent.sell(candle, self.agent.asset_capital)
                self.is_position_opened = False
                self.tp_target, self.sl_target = None, None
                action_executed = True
        if not action_executed:
            self.periods += 1
            self.agent.hold(candle)
        else:
            self.__reset_period(candle)
            self.drop_price = None
        return True

    def __reset_period(self, candle: Candle):
        self.periods = 0
        self.period_start_price = candle.close

    def __calculate_exit_targets(self, candle: Candle) -> tuple[float, float]:
        return candle.close * (1 + self.tp_pct), candle.close * (1 - self.sl_pct)
