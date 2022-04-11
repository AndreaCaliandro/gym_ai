from myenvs.trading_env import TradingEnv
import numpy as np

env = TradingEnv(stocks_list=('AMZN'),
                 initial_money=100000
                 )


def agent_action(stock_price, stock_owned, uninvested_cash, portfolio_amount):
    # return np.random.randint([10, 10, 10, 10, 10])
    return np.array([1])


for i_episode in range(2):
    print('New Episode')
    observation = env.reset()
    for t in range(10):
        env.render()
        print(observation)

        action = agent_action(**observation)
        print('Action', action)
        observation, reward, done, info = env.step(action)
        print(info)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()