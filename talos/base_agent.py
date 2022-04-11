from abc import abstractmethod
from gym import Env


class BaseAgent:
    """
    Abstract class of an agent.
    """

    def __init__(self, environment: Env):
        """

        :param environment: the context or board game with its rules where the agent take actions.
        """
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space

    @abstractmethod
    def action(self, *observation):
        """
        Decision making of the agent. This method is called every time the agent is request to take an action.
        By default this method return a random action among those allowed.
        """
        return self.action_space.sample()
