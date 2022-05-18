import gym
import numpy as np
from network import DQNAgent

# create a cartpole environment
env = gym.make('CartPole-v0')


# train the DQN agent in the cartpole environment
agent = DQNAgent(input_size=4, n_actions=2, hidden_size=16, gamma=0.99, batch_size=32, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

# initialize the agent's memory with random play
# reset the environment
state = env.reset()
# for each episode
i = 0
while agent.memory.len() < 1000:
    # render the environment
    #env.render()
    # select an action
    action = agent.get_action(state, agent.epsilon)
    # take the action
    next_state, reward, done, info = env.step(action)
    # add the transition to the replay memory
    agent.memory.push(state, action, reward, next_state, done)
    # update the state
    state = next_state
    # break if the episode is done
    #if done:
        # print("Episode finished after {} timesteps".format(i+1))
        # # print the length of the agent's memory
        # print("Length of agent's memory: {}".format(agent.memory.len()))
        # #break
    i += 1

for t in range(20):
    # reset the environment
    state = env.reset()
    # track the episode reward
    episode_reward = 0
    # for each episode
    for i in range(1000):
        # render the environment
        #env.render()
        # select an action
        action = agent.get_action(state, agent.epsilon)
        # take the action
        next_state, reward, done, info = env.step(action)
        # update the reward
        episode_reward += reward
        # add the transition to the replay memory
        agent.memory.push(state, action, reward, next_state, done)
        # update the DQN
        agent.update()
        # update the state
        state = next_state
        # break if the episode is done
        if done:
            print("Episode finished after {} timesteps".format(i+1))
            print("Episode reward: {}".format(episode_reward))
            break
