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
        self.internal_state = {'low_margin': 0,
                               'hi_margin': 0,
                               'ewm': 0,
                               'trade': 0,
                               'sold_stocks': np.array([]),
                               'bought_stocks': np.array([]),
                               'before_trade_stock_owned': np.array([])
                               }

    def action(self, stock_price, stock_memory, stock_owned, uninvested_cash, portfolio_amount, **kwargs):
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

        self.internal_state = {'low_margin': low_margin,
                               'hi_margin': hi_margin,
                               'ewm': ewm.iloc[-1],
                               'trade': trade,
                               'sold_stocks': sold_stocks,
                               'bought_stocks': bought_stocks,
                               'before_trade_stock_owned': stock_owned
                               }
        nso = stock_owned - sold_stocks + bought_stocks
        return nso

    def post_action_observation_update(self, trading_price_previous_action, average_stock_cost, **kwargs):
        average_stock_cost = average_stock_cost * self.internal_state['before_trade_stock_owned'] \
                             + self.internal_state['bought_stocks'] * trading_price_previous_action
        return {'average_stock_cost': average_stock_cost}