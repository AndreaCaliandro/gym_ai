import gym


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)


def agent_action(position, velocity, angle, angular_vel):
    if angular_vel<0:
        return 0
    else:
        return 1


for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)

        action = agent_action(*observation)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
