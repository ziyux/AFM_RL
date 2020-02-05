import csv
import numpy as np
from AFM_agent import Agent
from AFM_DQL import DQL
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
            if j == len(lastop[i]) - 1:
                break
            temp += ','
        feaname.append(temp)
    output = np.vstack((feaname, feature_space))
    SaveCSVfile(filename, output)
    return feaname


def read_data(filename, dataloc):
    file = OpenCSVfile(filename)
    tagname = np.array(file.x[0, :dataloc - 1])
    feaname = np.array(file.x[0, dataloc - 1:])
    # data = np.array(file.x[1:,:], dtype='float')
    # data = data[data[:,0].argsort()]
    tag = np.array(file.x[1:, :dataloc - 1])
    fea = np.array(file.x[1:, dataloc - 1:], dtype='float')

    res = tag[:, -1].reshape(len(tag))
    name = [res[0]]
    for i in range(len(res)):
        if res[i] in name:
            res[i] = int(name.index(res[i]))
        else:
            name.append(res[i])
            res[i] = int(name.index(res[i]))
    fea = fea[res.argsort()]
    res = np.array(np.sort(res), dtype=int)
    return tagname, feaname, tag, fea, res


def average_sliding_window(data, N):
    mylist = data
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    plt.title('Average rewards of every ' + str(N) + ' episodes')
    plt.xlabel('Episode')
    plt.plot(range(len(moving_aves)), moving_aves)
    plt.show()


# tagname, feaname, tag, fea, res = read_data('completedata.csv', 2)
tagname, feaname, tag, fea, res = read_data('testingdata.csv',4)
lastop = [['non', n] for n in feaname[:]]  # document the last operation
comp = list(np.zeros(len(feaname[:]), dtype=int))  # document each feature's complexity
lastphi = [0]  # document the shape of the previous phi to avoid repitetion when doing binary operation
feature_space = fea[:, :]

act = ['exp', 'sqr', 'div', 'sub', 'mul']
# act = ['exp', 'sum', 'p-1','p+2', 'div', 'sqr']
# act = ['exp', 'sum', 'p-1']
# act = ['exp', 'div', 'p+2']
env = Agent(fea, res, feaname, act, max_comp=3)
MAX_DIM = 1
learning = []
for d in range(MAX_DIM):
    episodes = 30
    exp_replay = True
    use_target = True
    score_list = []
    env.accuracy = [-1 for a in env.accuracy]
    learning.append(DQL(len(act), fea.shape[0], hidden_layer=(150, 120), batch_size=32, epsilon=1.0,
                        epsilon_decay=0.99, gamma=0.99, alpha=0.001, func_approximation='NN'))
    for i in range(episodes):
        state = env.reset(i)
        score = []
        while (True):
            action = learning[-1].greedy(env.features[:, state], env, act)
            reward, next_state, done = env.step(state, act[action])
            score.append(reward)
            learning[-1].fit(state, action, reward, next_state, done, env)
            state = next_state
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
    env.selected_set.append(selected)

    accuracy = env.compute_accuracy(env.selected_set, True)
    selected = np.array([x[1] for x in accuracy]).argsort()[-5:][::-1]
    for i in selected:
        print('Score: ', accuracy[i][1])
        print('Descriptor: ', [env.trace_descriptors(int(x)) for x in accuracy[i][0]])

    # Plot the rewards
    plt.title('Rewards of each episode')
    plt.xlabel('Episode')
    plt.plot(list(range(len(score_list))), score_list)
    plt.show()
    average_sliding_window(score_list, 10)
    print(env.features.shape)
output_feature_space(env.tag, env.features)
