import csv
import numpy as np
from agent import Agent
from DQL import DQL
import matplotlib.pyplot as plt


class OpenCSVfile(object):
    def __init__(self, filename):
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            x = []
            for row in csv_reader:
                x.append(row[:])
            self.x = np.array(x[:])


def SaveCSVfile(filename, data):
    with open(filename, mode='w', newline='') as csvfile:
        csv_witer = csv.writer(csvfile)
        for row in data:
            csv_witer.writerow(row)


def output_feature_space(lastop, feature_space, filename='feature_space.csv'):
    feaname = []
    for i in range(len(lastop)):
        temp = ''
        for j in range(len(lastop[i])):
            temp += str(lastop[i][j])
            if j == len(lastop[i])-1:
                break
            temp += ','
        feaname.append(temp)
    output = np.vstack((feaname, feature_space))
    SaveCSVfile(filename, output)
    return feaname


def read_data(filename, dataloc):
    file = OpenCSVfile(filename)
    tagname = np.array(file.x[0,:dataloc-1])
    feaname = np.array(file.x[0, dataloc - 1:])
    # data = np.array(file.x[1:,:], dtype='float')
    # data = data[data[:,0].argsort()]
    tag = np.array(file.x[1:,:dataloc - 1])
    fea = np.array(file.x[1:, dataloc - 1:], dtype='float')

    res = tag[:,-1].reshape(len(tag))
    name = [res[0]]
    for i in range(len(res)):
        if res[i] in name:
            res[i] = int(name.index(res[i]))
        else:
            name.append(res[i])
            res[i] = int(name.index(res[i]))
    fea = fea[res.argsort()]
    res = np.array(np.sort(res),dtype=int)
    return tagname, feaname, tag, fea, res


def average_sliding_window(data, N):
    mylist = data
    N = N
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    plt.title('Average rewards of every '+str(N)+' episodes')
    plt.xlabel('Episode')
    plt.plot(range(len(moving_aves)),moving_aves)
    plt.show()

# tagname, feaname, tag, fea, res = read_data('completedata.csv',2)
tagname, feaname, tag, fea, res = read_data('testingdata.csv',4)
lastop = [['non',n] for n in feaname[:]]    # document the last operation
comp = list(np.zeros(len(feaname[:]), dtype=int))   #document each feature's complexity
lastphi = [0]   #document the shape of the previous phi to avoid repitetion when doing binary operation
feature_space = fea[:,:]

# act = ['exp', 'sqr', 'div', 'sub', 'mul']
act = ['exp', 'sum', 'p-1','p+2', 'div', 'sqr']
# act = ['exp', 'div', 'p+2']
env = Agent(fea, res, feaname, act, max_comp=3)
learning = DQL(len(act), fea.shape[0], hidden_layer=(150, 120), batch_size=2, epsilon=1.0,
                 epsilon_decay = 0.99, gamma=0.99, alpha=0.001, func_approximation='NN')

episodes=10000
exp_replay=True
use_target=True
score_list = []
for i in range(episodes):
    state = env.reset(i)
    score = []
    while (True):
        action = learning.greedy(env.features[:, state])
        reward, next_state, done = env.step(state, act[action])
        score.append(reward)
        if exp_replay:  # control the experience replay on and off
            learning.exp_buffer.append([state, action, reward, next_state, done])
            state = next_state
            learning.replay(env.features)
        else:
            if use_target:  # control the target network on and off
                learning.target_network.append([state, action, reward, next_state, done])
                state = next_state
                learning.target_network_fit()
            else:
                Qtarget = reward + learning.gamma * (np.amax(learning.model.predict(next_state), axis=1)) * (1 - done)
                Qvalue = learning.model.predict(state)
                Qvalue[0, action] = Qtarget
                learning.model.fit(state, Qvalue, epochs=1, verbose=0)
                state = next_state
                if learning.epsilon > learning.epsilon_min: learning.epsilon *= learning.epsilon_decay

        if done:
            print("Episode: {}  Score: {}".format(i + 1, max(score)))
            break
    score_list.append(max(score))
    complete = max(score_list)
    if complete > 0.999999:
        print('Training has completed')
        break
    print("Max score: {} \n".format(complete))
selected = np.array(env.accuracy).argsort()[-5:][::-1]
for i in selected:
    print('Score: ', env.accuracy[i])
    print('Descriptor: ', env.trace_descriptors(int(i)))

#Plot the rewards
plt.title('Rewards of each episode')
plt.xlabel('Episode')
plt.plot(list(range(len(score_list))), score_list)
plt.show()
average_sliding_window(score_list, 200)
print(env.features.shape)
