import numpy as np

from scipy.stats import beta

from talos.base_agent import BaseAgent


class DumbAgent(BaseAgent):

    def action(self, observation, **kwargs):
        """
        Select randomly one of the stocks
        """
        print('action')
        print(self.action_space.shape[0])
        stock_owned = np.zeros(self.action_space.shape[0])
        stock_index = np.random.randint(self.action_space.shape[0])
        stock_owned[stock_index] = 1
        print(stock_owned)
        print('end action')
        return stock_owned


class BayesianAgent(BaseAgent):
    """
    Implement the Thompson Sampling Theory
    """
    def __init__(self, environment):
        super(BayesianAgent, self).__init__(environment)
        self.n_stocks = environment.n_stocks
        self.internal_state = {
            'alpha': np.array([1.0] * self.n_stocks),
            'beta': np.array([1.0] * self.n_stocks),
            'num_pulls': np.array([0] * self.n_stocks)
        }
        self.internal_state.update({'p_estimate': beta.stats(self.internal_state['alpha'],
                                                             self.internal_state['beta'],
                                                             moments='m'
                                                             )})

    def internal_state_update(self, outcome, slot_index):
        self.internal_state['alpha'][slot_index] += outcome
        self.internal_state['beta'][slot_index] += 1 - outcome
        self.internal_state['num_pulls'][slot_index] += 1
        self.internal_state['p_estimate'][slot_index] = beta.stats(self.internal_state['alpha'][slot_index],
                                                                   self.internal_state['beta'][slot_index],
                                                                   moments='m'
                                                                   )
        self.internal_state.update()

    def sample(self, slot_index):
        a = self.internal_state['alpha'][slot_index]
        b = self.internal_state['beta'][slot_index]
        return np.random.beta(a, b)

    def action(self, observation, **kwargs):
        return np.argmax([self.sample(j) for j in range(self.n_stocks)])
