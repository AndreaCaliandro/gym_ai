import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import yfinance as yf


class TradingEnv(gym.Env):

    metadata = {"render.modes": [] }
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, stocks_list=(), initial_money=None, risk_factor = (1.0, 1.0),
                 full_history_start='2010-01-01', full_history_end='2022-02-01'):

        self.n_stocks = len(stocks_list)
        self.action_space = spaces.Box(np.array([0]*self.n_stocks), np.array([np.inf]*self.n_stocks), dtype=np.int)
        self.observation_space = spaces.Dict({
            "stock_price": spaces.Box(low=0, high=np.inf, shape=(self.n_stocks, ), dtype=np.float32),
            "stock_owned": spaces.Box(low=0, high=np.inf, shape=(self.n_stocks, ), dtype=np.int),
            "uninvested_cash": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            "portfolio_amount": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32)})

        self.initial_money = initial_money  # Moneys available at the beginning of each episode
        self.risk_factor = risk_factor   # Determine how much we are sensitive to losses or gains, respectively.
                                         # It is a weight factor applied in teh calculation of the reward
        self.full_history = self.scrape_stock_history(stocks_list, full_history_start, full_history_end)
        print('Full history len', len(self.full_history))
        self.stock_history = None
        self.trading_day = 0
        self.total_reward = 0

        # The variables below are set at the end of each step and represent the step state
        self.stock_price = None  # Array of stock price of each firm
        self.stock_owned = None  # Array with Number of stocks of each firm
        self.uninvested_cash = None  # Residual cash not invested in stocks
        self.portfolio_amount = None  # Total available amount of moneys
        self.state = {}

    def scrape_stock_history(self, stock_list, start, end):
        data_df = yf.download(stock_list,start,end)
        return data_df['Open'].values

    def reset(self, seed=0):
        if self.initial_money is None:
            self.initial_money = np.random.random() * 1000000
        self.stock_owned = np.array([0]*self.n_stocks)
        initial_day = np.random.randint(len(self.full_history)-52*5)
        self.stock_history = self.full_history[initial_day: initial_day+52*5]
        print('Stock history len', len(self.stock_history))
        self.trading_day = 0
        self.stock_price = self.get_stock_price()
        self.portfolio_amount = self.initial_money
        self.uninvested_cash = self.initial_money
        self.state = {
            "stock_price": self.stock_price,
            "stock_owned": self.stock_owned,
            "uninvested_cash": self.uninvested_cash,
            "portfolio_amount": self.portfolio_amount
        }
        return self.state

    def step_reward(self, current_stock_price):
        portfolio_delta = (current_stock_price - self.stock_price) * self.stock_owned
        reward = 0
        for delta in portfolio_delta:
            if delta < 0:
                reward += delta * self.risk_factor[0]
            else:
                reward += delta * self.risk_factor[1]
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
        self.trade(action)
        done = False
        if self.uninvested_cash < 0:
            done = True

        self.trading_day += 1
        current_stock_price = self.get_stock_price()
        if done:
            reward = -np.inf
        else:
            reward = self.step_reward(current_stock_price)

        self.total_reward += reward
        self.stock_price = current_stock_price
        self.portfolio_amount = (current_stock_price * self.stock_owned).sum() + self.uninvested_cash

        self.state = {
            "stock_price": self.stock_price,
            "stock_owned": self.stock_owned,
            "uninvested_cash": self.uninvested_cash,
            "portfolio_amount": self.portfolio_amount
        }

        info = {
            "gain": self.portfolio_amount - self.initial_money,
            "reward": reward,
            "total_reward": self.total_reward
        }

        return self.state, reward, done, info

    def trade(self, action):
        self.stock_owned = action
        self.uninvested_cash = self.portfolio_amount - (self.stock_price * self.stock_owned).sum()

    def get_stock_price(self):
        return self.stock_history[self.trading_day]

    def render(self, mode='human'):
        return 0
