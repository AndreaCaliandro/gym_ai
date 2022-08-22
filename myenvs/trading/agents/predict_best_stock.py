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


class EmaAgent(BaseAgent):
    """
    Each day calculate the EMA af each stock change, to model them with gaussian with fixed variance.
    Point all on the stock with highest change calculated sampling a value from their models.
    The day after check that the stock pointed to had effectively the highest change.
    If true, reduce the variance of its model by a delta.
    If false, increase its variance by a delta.
    """
    def __init__(self, environment, initial_sigma=1.0, ema_alpha=0.1):
        super(EmaAgent, self).__init__(environment)
        self.n_stocks = environment.n_stocks
        self.sigma0 = initial_sigma
        self.ema_alpha = ema_alpha

    def reset(self):
        self.internal_state = {
            'change_mean': self.environment.stock_change,
            'change_sigma': np.array([self.sigma0] * self.n_stocks),
            'num_pulls': np.zeros(self.n_stocks)
        }

    def exp_moving_ave(self, emas, values):
        return values * self.ema_alpha + emas * (1 - self.ema_alpha)

    def internal_state_update(self, stock_owned, stock_change, **kwargs):
        self.internal_state['change_mean'] = self.exp_moving_ave(self.internal_state['change_mean'], stock_change)

        stock_index = np.argmax(stock_owned)
        if np.argmax(stock_change) == stock_index:
            self.internal_state['num_pulls'][stock_index] += 1
        else:
            self.internal_state['num_pulls'][stock_index]  -= 0.5
            self.internal_state['num_pulls'][stock_index] = max(0, self.internal_state['num_pulls'][stock_index])
        self.internal_state['change_sigma'] = \
            self.internal_state['change_sigma'] / np.sqrt(self.internal_state['num_pulls'] + 1)

    def sample(self):
        mean = np.array(self.internal_state['change_mean'], dtype=float)
        sigma = self.internal_state['change_sigma']
        return np.random.normal(mean, sigma)

    def action(self, observation, **kwargs):
        stock_owned = np.zeros(self.action_space.shape[0])
        stock_index = np.argmax(self.sample())
        stock_owned[stock_index] = 1  # TODO Do not buy just one. Instead, use the full portfolio to buy this stock
        return stock_owned
