import random
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.activations import relu, linear
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DQL:
    def __init__(self, action_space, state_space, hidden_layer=(150, 120), batch_size=64, epsilon=1.0,
                 epsilon_decay = 0.996, gamma=0.99, alpha=0.001, func_approximation='NN'):
        self.act_space = action_space
        self.stat_space = state_space
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.hidden_layer = hidden_layer
        self.func_approximation = func_approximation
        self.exp_buffer = []
        self.target_network = []
        self.model = self.construct_model()
        np.random.seed(0)

    def construct_model(self):
        model = Sequential()
        if self.func_approximation == 'NN':               #control the function approximator we will use
            model.add(Dense(self.hidden_layer[0], input_dim=self.stat_space, activation=relu))
            for i in range(len(self.hidden_layer) - 1):
                model.add(Dense(self.hidden_layer[i + 1], activation=relu))
            model.add(Dense(self.act_space, activation=linear))
            model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        if self.func_approximation == 'Linear':
            model.add(Dense(self.act_space, input_dim=self.stat_space, activation=linear))
            model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        return model

    def greedy(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.act_space)
        else:
            # preprocessing
            normalization = MinMaxScaler()
            state = normalization.fit_transform(state.reshape(-1, 1)).reshape(1,-1)
            # print(state.shape)
            # print('predict',self.model.predict([state]))
            action = np.argmax(self.model.predict(state)[0])
            # print('action',action)
        return action

    def replay(self, features):

        if len(self.exp_buffer) < self.batch_size: return
        minibatch = random.sample(self.exp_buffer, self.batch_size)
        states = np.squeeze(np.array([features[:, i[0]] for i in minibatch]))
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.squeeze(np.array([features[:, i[3]] for i in minibatch]))
        dones = np.array([i[4] for i in minibatch])

        # print(states)
        # print(next_states)
        #preprocessing
        normalization = MinMaxScaler()
        for i in range(len(states)):
            states[i] = normalization.fit_transform(states[i].reshape(-1, 1)).reshape(1, -1)
        for i in range(len(next_states)):
            next_states[i] = normalization.fit_transform(next_states[i].reshape(-1, 1)).reshape(1, -1)

        Qtargets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1 - dones)
        Qvalue = self.model.predict_on_batch(states)
        Qvalue[[i for i in range(self.batch_size)], [actions]] = Qtargets
        # print('targets',Qtargets)
        # print('qvalue',Qvalue)
        self.model.fit(states, Qvalue, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def target_network_fit(self):
        if len(self.target_network) < self.batch_size: return
        states = np.squeeze(np.array([i[0] for i in self.target_network]))
        actions = np.array([i[1] for i in self.target_network])
        rewards = np.array([i[2] for i in self.target_network])
        next_states = np.squeeze(np.array([i[3] for i in self.target_network]))
        dones = np.array([i[4] for i in self.target_network])
        Qtargets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        Qvalue = self.model.predict_on_batch(states)
        Qvalue[[i for i in range(self.batch_size)], [actions]] = Qtargets
        self.model.fit(states, Qvalue, epochs=1, verbose=0)
        self.target_network = []
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    # def train(self, episodes=400, environment='LunarLander-v2', direction='dqn-agent-results', exp_replay=True, use_target=True):
    #     score_list = []
    #     env = gym.make(environment)
    #     env = wrappers.Monitor(env, direction, force=True)
    #     env.seed(0)
    #     for i in range(episodes):
    #         state = env.reset()
    #         state = np.reshape(state, (1, 8))
    #         score = 0
    #         while(True):
    #             action = self.greedy(state)
    #             env.render()
    #             next_state, reward, done, info = env.step(action)
    #             next_state = np.reshape(next_state, (1, 8))
    #             score += reward
    #             if exp_replay:                      #control the experience replay on and off
    #                 self.exp_buffer.append([state, action, reward, next_state, done])
    #                 state = next_state
    #                 self.replay()
    #             else:
    #                 if use_target:                  #control the target network on and off
    #                     self.target_network.append([state, action, reward, next_state, done])
    #                     state = next_state
    #                     self.target_network_fit()
    #                 else:
    #                     Qtarget = reward + self.gamma * (np.amax(self.model.predict(next_state), axis=1)) * (1 - done)
    #                     Qvalue = self.model.predict(state)
    #                     Qvalue[0, action] = Qtarget
    #                     self.model.fit(state, Qvalue, epochs=1, verbose=0)
    #                     state = next_state
    #                     if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    #
    #             if done:
    #                 print("Episode: {}  Score: {}".format(i+1, score))
    #                 break
    #         score_list.append(score)
    #         complete = np.mean(score_list[-60:])
    #         if complete > 200:
    #             print('Training has completed')
    #             break
    #         print("Average score: {} \n".format(complete))
    #     env.close()
    #     return score_list


