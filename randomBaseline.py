import gym
import numpy as np

# create a CartPole environment
env = gym.make('CartPole-v0')

# test CartPole environment with a random agent
for i_episode in range(20):
    observation = env.reset()
    # track the sum of rewards across timesteps for each episode
    cumulative_reward = 0
    for t in range(100):
        #env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # print cumulative reward for each episode
            print("Cumulative reward: {}".format(cumulative_reward))
            break
