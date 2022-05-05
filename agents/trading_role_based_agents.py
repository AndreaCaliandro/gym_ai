import numpy as np

from myenvs.trading_env import TradingEnv
from talos.base_agent import BaseAgent
from agents.trading.trend import trend_margins, exponential_moving_average, daily_volatility


class DummyAgent(BaseAgent):

    def action(self, **observation):
        return np.array([1]*self.action_space.shape[0])


class OneStock(BaseAgent):

    def __init__(self, environment, stock_name, window_size):
        super(OneStock, self).__init__(environment)
        self.stock_name = stock_name
        self.window_size = window_size

    def action(self, stock_price, stock_memory, stock_owned, uninvested_cash, portfolio_amount):
        # print("OneStock agent action")
        margin = trend_margins(stock_memory, self.window_size)
        timeseries_df = stock_memory[['Open']].append({'Open': stock_price}, ignore_index=True)
        ewm = exponential_moving_average(timeseries_df, self.window_size)[f'ewm_{self.window_size}']

        low_margin = ewm.iloc[-1] * (1- margin)
        hi_margin = ewm.iloc[-1] * (1 + margin)

        sold_stocks = stock_owned * 0
        bought_stocks = stock_owned * 0

        if stock_price > hi_margin:
            # Sell
            trade = -1
            sold_stocks = stock_owned - np.floor(stock_owned/2)
        elif stock_price < low_margin:
            # Buy
            trade = 1
            bought_stocks = np.array([np.floor(uninvested_cash * 0.5 / stock_price)])
        else:
            trade = 0

        internal_state = {'low_margin': low_margin,
                          'hi_margin': hi_margin,
                          'ewm': ewm.iloc[-1],
                          'trade': trade,
                          'sold_stocks': sold_stocks,
                          'bought_stocks': bought_stocks
                         }
        nso = stock_owned - sold_stocks + bought_stocks
        return nso, internal_state
