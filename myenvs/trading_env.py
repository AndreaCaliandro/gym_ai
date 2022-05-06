import gym
from gym import spaces
from gym.envs.registration import EnvSpec

import numpy as np
import yfinance as yf


class TradingEnv(gym.Env):
    """
    Each day (or step), we initially know the Open price of the stocks in the list,
    and the stock stats (Open, High, Low, CLose) of the previous N days.
    On teh base of it we decide whether to buy or sell.
    But we sell at a price different than the Open. We can cofigure the selling price to be
    the Close, Low, High, or the Day Average.
    """
    metadata = {"render.modes": [] }
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, stocks_list=(), initial_money=None, risk_factor = (1.0, 1.0),
                 full_history_start='2010-01-01', full_history_end='2022-02-01',
                 stock_memory_lenght=5*13,  # About 3 months
                 episode_lenght=5*52,  # One year
                 trading_price_column='Close'
                ):

        self.spec = EnvSpec(id='Trading-v0', max_episode_steps=episode_lenght)

        self.n_stocks = len(stocks_list)
        print('len stocks list', self.n_stocks)
        self.action_space = spaces.Box(np.array([0]*self.n_stocks), np.array([np.inf]*self.n_stocks), dtype=np.int)
        self.observation_space = spaces.Dict({
            "stock_price": spaces.Box(low=0, high=np.inf, shape=(self.n_stocks, ), dtype=np.float32),
            "stock_memory": spaces.Space(),
            "stock_owned": spaces.Box(low=0, high=np.inf, shape=(self.n_stocks, ), dtype=np.int),
            "uninvested_cash": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            "portfolio_amount": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "trading_price_previous_action": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "average_stock_cost": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
        })

        self.initial_money = initial_money  # Moneys available at the beginning of each episode
        self.risk_factor = risk_factor   # Determine how much we are sensitive to losses or gains, respectively.
                                         # It is a weight factor applied in the calculation of the reward
        self.trading_price_column = trading_price_column
        self.full_history = self.scrape_stock_history(stocks_list, full_history_start, full_history_end)
        print('Full history len', len(self.full_history))
        self.memory_lenght = stock_memory_lenght
        self.episode_lenght = episode_lenght

        self.stock_history = None
        self.trading_day = 0
        self.total_reward = 0

        # The variables below are set at the end of each step and represent the step state
        self.stock_price = None  # Array of stock price of each firm
        self.stock_owned = None  # Array with Number of stocks of each firm
        self.uninvested_cash = None  # Residual cash not invested in stocks
        self.portfolio_amount = None  # Total available amount of moneys
        self.state = {}
        self.market_value = None

    def scrape_stock_history(self, stock_list, start, end):
        data_df = yf.download(stock_list,start,end)
        return data_df[['Open','Close','Low','High']].reset_index()

    def reset(self, seed=0):
        if self.initial_money is None:
            self.initial_money = np.random.random() * 1000000

        self.stock_owned = np.array([0]*self.n_stocks)

        initial_day = np.random.randint(len(self.full_history) - self.episode_lenght)
        self.stock_history = self.full_history.iloc[initial_day: initial_day + self.episode_lenght]
        print('Stock history len', len(self.stock_history))

        self.trading_day = initial_day
        self.portfolio_amount = self.initial_money
        self.uninvested_cash = self.initial_money
        self.stock_price = self.get_stock_price()
        self.stock_memory = self.update_stock_memory()
        self.total_reward = 0
        self.state = {
            "stock_price": self.stock_price,
            "stock_memory": self.stock_memory,
            "stock_owned": self.stock_owned,
            "uninvested_cash": self.uninvested_cash,
            "portfolio_amount": self.portfolio_amount,
            "trading_price_previous_action": 0,
            "average_stock_cost": np.array([0]*self.n_stocks)
        }
        self.market_value = self.stock_price.sum()
        return self.state

    def step_reward(self, traded_portfolio):
        """
        Reward is calculated as gain respect to the initial aomunt of money invested
        """
        delta = traded_portfolio/self.initial_money - 1
        if delta < 0:
            reward = delta * self.risk_factor[0]
        else:
            reward = delta * self.risk_factor[1]
        return reward

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        trading_price = self.get_trading_price()
        traded_portfolio = self.trade(action)  # In this moment only the Open price of the day is known

        done = False
        reward = self.step_reward(traded_portfolio)
        if self.uninvested_cash < 0:
            reward = -np.inf
            done = True
            print('OUT OF CASH!!!')
        self.total_reward += reward

        self.trading_day += 1
        if self.trading_day == self.stock_history.index[-1]:
            done = True

        self.stock_price = self.get_stock_price()
        self.portfolio_amount = self.vested_amount(self.stock_price) + self.uninvested_cash
        self.stock_memory = self.update_stock_memory()

        self.state.update({
            "stock_price": self.stock_price,
            "stock_memory": self.stock_memory,
            "stock_owned": self.stock_owned,
            "uninvested_cash": self.uninvested_cash,
            "portfolio_amount": self.portfolio_amount,
            "trading_price_previous_action": trading_price,
        })

        info = {
            "gain": self.portfolio_amount - self.initial_money,
            "gain_percent": 100 * (self.portfolio_amount/self.initial_money - 1),
            "reward": reward,
            "total_reward": self.total_reward
        }

        return self.state, reward, done, info

    def trade(self, action):
        trading_price = self.get_trading_price()
        sell_all = self.vested_amount(trading_price)
        self.stock_owned = action
        self.uninvested_cash += sell_all - self.vested_amount(trading_price)
        return self.uninvested_cash + self.vested_amount(trading_price)

    def get_stock_price(self):
        return self.stock_history.loc[self.trading_day, 'Open']

    def update_stock_memory(self):
        return self.full_history.iloc[self.trading_day - self.memory_lenght: self.trading_day]

    def vested_amount(self, stock_price):
        return (stock_price * self.stock_owned).sum()

    def get_trading_price(self):
        return self.full_history.loc[self.trading_day, self.trading_price_column]

    def render(self, mode='human'):
        return 0
