import pandas as pd

from talos.learning_loop import LearningLoop
from myenvs.trading.trading_env import TradingEnv
from agents.trading_role_based_agents import OneStock
from agents.predict_best_stock import DumbAgent
from myenvs.trading.rewards.market_reward import MarketReward, BestStock, GainReward

import warnings


warnings.filterwarnings("ignore")

reward_class = GainReward(loss_risk_factor=1.0, gain_risk_factor=1.0)
# reward_class = BestStock()

env = TradingEnv(stocks_list=('NVDA',), #'AMZN', 'AAPL', 'NKE', 'T'),
                 reward_class=reward_class,
                 initial_money=100000,
                 stock_memory_length=5*10,  # 10 weeks
                 episode_length=5*26,  # 1 year
                 )

agent = OneStock(environment=env, stock_name='NVDA', window_size=5)
# agent = DumbAgent(environment=env)


class TradingLoop(LearningLoop):

    def __init__(self, **kwargs):
        super(TradingLoop, self).__init__(**kwargs)
        self.stats.update(gain=[],
                          gain_percent=[],
                          total_reward=[],
                          cash=[],
                          portfolio_amount=[],
                          market_change=[])

    def additional_stats(self, info):
        self.stats['gain'].append(info['gain'])
        self.stats['gain_percent'].append(info['gain_percent'])
        self.stats['total_reward'].append(info['total_reward'])
        self.stats['cash'].append(self.env.uninvested_cash)
        self.stats['portfolio_amount'].append(self.env.portfolio_amount)
        self.stats['market_change'].append(100 * (self.env.stock_price.sum()/self.env.market_value - 1))


if __name__ == '__main__':

    learn = TradingLoop(environment=env, agent=agent, episodes=10)
    learn.time_limit_loop()

    learn.plot_stats('portfolio_amount').show()
    learn.plot_stats('cash').show()
    learn.plot_stats('total_reward').show()
    fig = learn.plot_stats('gain_percent')
    learn.plot_stats('market_change', figure=fig).show()

    records = [{key:d[key] for key in d if key!='stock_memory'} for d in learn.log_records]
    df = pd.DataFrame(records)
    df.to_json('../logs/trading_3.json')
