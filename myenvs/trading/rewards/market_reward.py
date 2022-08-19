import numpy as np

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
