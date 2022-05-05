from gym import Env
from gym.wrappers.time_limit import TimeLimit
from talos.base_agent import BaseAgent
from matplotlib import pyplot as plt


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
        self.stats = {'episode_num': [],
                      'steps': []
                      }
        self.log_records = []

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

                record = {'episode': episode, 'step': t}
                record.update(observation)

                action = self.agent.action(**observation)
                record.update(self.agent.internal_state)

                observation, reward, done, info = self.env.step(action)
                observation.update(self.agent.post_action_observation_update(**observation))

                record.update(info)
                self.log_records.append(record)

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    self.stats['episode_num'].append(episode)
                    self.stats['steps'].append(t+1)
                    self.additional_stats(info)
                    break
        self.env.close()

    def additional_stats(self, info):
        pass

    def plot_stats(self, stats_name='steps', figure=None):
        if figure is None:
            figure = plt.figure()
        plt.plot(self.stats['episode_num'], self.stats[stats_name], label=stats_name)
        plt.title(stats_name)
        plt.ylabel(stats_name)
        plt.xlabel('Episode')
        plt.legend()
        return figure
