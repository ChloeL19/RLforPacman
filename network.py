import torch
import numpy as np
import random


# create a Deep Q Network with experience replay memory
class DQN:
    def __init__(self, input_size, n_actions, hidden_size, gamma):
        # define the neural network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_actions)
        )
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # define the loss function
        self.loss_fn = torch.nn.MSELoss()
        # define the discount factor
        self.gamma = gamma
        self.n_actions = n_actions

    def state_dict(self):
        return self.model.state_dict()   

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def update(self, memory, batch_size, target_net):
        # sample a batch of transitions from the memory
        #print(zip(*memory.sample(batch_size)))
        # print contents of zip(*memory.sample(batch_size))
        #print(list(zip(*memory.sample(batch_size))))
        #state, action, reward, next_state, done = list(zip(*memory.sample(batch_size)))
        memorybatch = list(zip(*memory.sample(batch_size)))
        #state, action, reward, next_state, done = memorybatch[:][0], memorybatch[:][1], memorybatch[:][2], memorybatch[:][3], memorybatch[:][4]
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for element in memorybatch:
            state.append(element[0])
            action.append(element[1])
            reward.append(element[2])
            next_state.append(element[3])
            done.append(element[4])
        # convert the list of transitions to tensors
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)
        # compute the q-values
        q_values = self.forward(state)
        next_q_values = self.forward(next_state)
        # compute the target q-values
        # converte done to integer values
        #print("done type: {}".format(done.type()))
        #print(done)
        next_q_values[done] = 0.0
        target_q_values = reward + self.gamma * next_q_values.max(1)[0]
        # compute the loss
        loss = self.loss_fn(q_values.gather(1, action.unsqueeze(1)), target_q_values.unsqueeze(1))
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update the target network
        target_net.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        # select an action using epsilon-greedy policy
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.forward(torch.tensor(state, dtype=torch.float)).detach().numpy()
            return np.argmax(q_values)


# create a replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        self.memory.append(args)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # debug
        return zip(*random.sample(self.memory, batch_size))
    
    def len(self):
        return len(self.memory)


# create a Deep Q Network agent with experience replay for the CartPole-v0 environment
class DQNAgent:
    def __init__(self, input_size, n_actions, hidden_size, gamma, batch_size, epsilon, epsilon_min, epsilon_decay):
        # define the hyperparameters
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # create the replay memory
        self.memory = ReplayMemory(capacity=1000)
        # create the DQN
        self.dqn = DQN(input_size, n_actions, hidden_size, gamma)
        # create the target DQN
        self.target_dqn = DQN(input_size, n_actions, hidden_size, gamma)
        # initialize the target DQN
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def update(self):
        # sample a batch of transitions from the memory
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # update the DQN
        self.dqn.update(self.memory, self.batch_size, self.target_dqn)
        # update the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state, epsilon):
        # select an action using epsilon-greedy policy
        return self.dqn.get_action(state, epsilon)

    def push(self, *args):
        self.memory.push(*args)