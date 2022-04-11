from gym import Env
from gym.wrappers.time_limit import TimeLimit
from talos.base_agent import BaseAgent
from matplotlib import pyplot as plt
import numpy as np


class LearningLoop:
    """
    Train the reinforcement learning machine looping over many episodes in the given environment.
    """

    def __init__(self, environment: Env, agent: BaseAgent, episodes=100):
        """

        :param environment: An instance of gym.Env. Context or board game with its rules where the agent take actions.
        :param agent: instance of a class derived by talos.BaseAgent.
        :param episodes: number of episodes the agent is trained on.
        """
        self.env = environment
        self.agent = agent
        self.episodes = episodes
        self.stats = []

    def training(self):
        if isinstance(self.env, TimeLimit):
            self.time_limit_loop()
        else:
            raise NotImplementedError('Training for environment of type %s not implemented' % type(self.env))

    def time_limit_loop(self):
        for episode in range(self.episodes):
            observation = self.env.reset()
            for t in range(self.env.spec.max_episode_steps):
                self.env.render()
                # print(observation)

                action = self.agent.action(*observation)
                observation, reward, done, info = self.env.step(action)

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    self.stats.append([episode, t+1])
                    break
        self.env.close()

    def plot_stats(self):
        fig = plt.figure()
        stats = np.array(self.stats)
        plt.plot(stats[:, 0], stats[:, 1])
        fig.show()
