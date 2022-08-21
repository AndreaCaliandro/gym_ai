import numpy as np
import pandas as pd

from talos.base_reward import BaseReward


class MarketReward(BaseReward):
    """
    Reward calculate against an investor that uses all his budget
    to buy stocks at the beginning of the trading period.
    The reward is proportional to how much the agent outperform such investor
    """

    def step_reward(self, traded_portfolio, stock_price=np.ones(1), market_value=1, initial_money=1, **status):
        virtual_agent_gain = stock_price.sum() / market_value - 1
        agent_gain = traded_portfolio / initial_money - 1
        return agent_gain - virtual_agent_gain


class GainReward(BaseReward):
    """
    Reward is calculated as gain respect to the initial amount of money invested
    """

    def __init__(self, loss_risk_factor, gain_risk_factor):
        super(GainReward, self).__init__()
        self.risk_factor = [loss_risk_factor, gain_risk_factor]

    def step_reward(self, traded_portfolio, initial_money=100, **status):
        delta = traded_portfolio/initial_money - 1
        if delta < 0:
            reward = delta * self.risk_factor[0]
        else:
            reward = delta * self.risk_factor[1]
        return reward


class BestStock(BaseReward):
    """
    Get rewarded if capable to predict which of the stocks has the highest gain
    """

    def step_reward(self, traded_portfolio, stock_owned=np.zeros(1),
                    stock_price=np.ones(1), trading_price_previous_action=np.ones(1), **status):
        """
        We assume that the agent posts all the portfolio on the stock predicted to have the highest gain
        """
        stock_variation = stock_price / trading_price_previous_action
        print(stock_owned)
        print(stock_variation)
        return np.argmax(stock_owned) == np.argmax(stock_variation)
