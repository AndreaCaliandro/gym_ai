import gym
import numpy as np

from talos.learning_loop import LearningLoop
from talos.base_agent import BaseAgent

env = gym.make('CartPole-v1')


class CartPoleAgent(BaseAgent):

    def __init__(self, env):
        super().__init__(env)

    def action(self, position, velocity, angle, angular_vel):
        if angular_vel<-0.094:
            action = 0
        elif angular_vel>0.094:
            action = 1
        elif velocity>0:
            action = 1
        else:
            action = 0
        return action


if __name__ == '__main__':
    agent = CartPoleAgent(env)
    learn = LearningLoop(env, agent)

    learn.training()
    learn.plot_stats()
    print((np.array(learn.stats)[:, 1] >= 475).sum())
