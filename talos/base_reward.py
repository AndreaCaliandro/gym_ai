from abc import abstractmethod


class BaseReward:
    """Compute the reward given the outcome of the agent action"""

    def __init__(self, **kwargs):
        self.total_reward = 0

    @abstractmethod
    def step_reward(self, traded_portfolio, **status):
        """Reward obtained after a single action"""
        pass

    def update_total_reward(self, reward, **kwargs):
        """Cumulative reward update"""
        self.total_reward += reward
