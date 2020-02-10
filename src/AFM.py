import csv
import numpy as np
from AFM_agent import Agent
from AFM_DQL import DQL
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class AFM(object):
    def __init__(self, filename, dataloc, act, mode, episodes=100, max_comp=3, dim=1):
        self.mode = mode
        self.dim = dim
        self.act = act
        self.episodes = episodes
        self.learning = []
        self.score_list = []
        self.increase_dim = False
        self.increase_dim_num = 1
        self.sample_name, self.fea_name, self.sample, self.fea, self.target = self.read_data(filename, dataloc)
        self.env = Agent(self.fea, self.target, self.fea_name, self.act, self.mode, max_comp)

    def open_csv_file(self, filename):
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            data = []
            for row in csv_reader:
                data.append(row[:])
            return np.array(data[:])

    def save_csv_file(self, filename, data):
        with open(filename, mode='w', newline='') as csvfile:
            csv_witer = csv.writer(csvfile)
            for row in data:
                csv_witer.writerow(row)

    def read_data(self, filename, dataloc):
        dataset = self.open_csv_file(filename)
        sample_name = np.array(dataset[0, :dataloc - 1])
        fea_name = np.array(dataset[0, dataloc - 1:])
        # data = np.array(dataset[1:,:], dtype='float')
        # data = data[data[:,0].argsort()]
        sample = np.array(dataset[1:, :dataloc - 1])
        fea = np.array(dataset[1:, dataloc - 1:], dtype='float')

        target = sample[:, -1].reshape(len(sample))
        if self.mode == 1:
            name = [target[0]]
            for i in range(len(target)):
                if target[i] in name:
                    target[i] = int(name.index(target[i]))
                else:
                    name.append(target[i])
                    target[i] = int(name.index(target[i]))
            fea = fea[target.argsort()]
            target = np.array(np.sort(target), dtype=int)
        else:
            target = np.array(target, dtype=float)
        return sample_name, fea_name, sample, fea, target

    def output_feature_space(self, sample_name, fea):
        fea_name = []
        for i in range(len(sample_name)):
            temp = ''
            for j in range(len(sample_name[i])):
                temp += str(sample_name[i][j])
                if j == len(sample_name[i]) - 1:
                    break
                temp += ','
            fea_name.append(temp)
        output = np.vstack((fea_name, fea))
        self.save_csv_file(str(self.dim)+'D_feature_space', output)
        return fea_name

    def average_sliding_window(self, data, N):
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

    def new_loop(self, iteration):
        if self.increase_dim:
            selected = np.array(self.env.reward).argsort()[-5:][::-1]
            for i in range(self.increase_dim_num):
                self.env.selected_set.append(selected)
            self.env.reward = [-1 for a in self.env.reward]
            self.dim += self.increase_dim_num
            self.score_list = []
        if self.increase_dim | (iteration == 0):
            self.learning.append(DQL(len(self.act), self.fea.shape[0], hidden_layer=(150, 120), batch_size=32,
                                     epsilon=1.0, epsilon_decay=0.99, gamma=0.99, alpha=0.001, func_approximation='NN'))

    def loop_control(self, action=None):
        if action is None:
            action = [input('Choose an action to continue:\nY: continue another loop without change'
                                        '\nC: change some parameters to continue\nQ: quit the program\n')]
        if (action[0] == 'Y') | (action[0] == 'y') | (action[0] == ''):
            self.increase_dim = False
            return False
        elif (action[0] == 'C') | (action[0] == 'c'):
            if len(action) < 2:
                action.append(input('Enter new episodes:\n'))
            if action[1] > '0':
                eng.episodes = int(action[1])
            if len(action) < 3:
                action.append(input('Enter number of dimension to increase:\n'))
            if action[2] > '0':
                self.increase_dim = True
                self.increase_dim_num = int(action[2])
            else:
                self.increase_dim = False
            return False
        elif (action[0] == 'Q') | (action[0] == 'q'):
            return True
        else:
            print('Warning: invalid input\n')
            return self.loop_control()

    def main(self, draw=False):

        for i in range(self.episodes):
            state = self.env.reset(i)
            score = []
            while (True):
                action = self.learning[-1].greedy(self.env.features[:, state], self.env, self.act)
                reward, next_state, done = self.env.step(state, self.act[action])
                score.append(reward)
                self.learning[-1].fit(state, action, reward, next_state, done, self.env)
                state = next_state
                if done:
                    print("Episode: {}  Score: {}".format(len(self.score_list) + 1, max(score)))
                    break
            self.score_list.append(max(score))
            complete = max(self.score_list)
            if complete > 0.999999:
                print('Training has completed')
                break
            print("Max score: {} \n".format(complete))
        selected = np.array(self.env.reward).argsort()[-5:][::-1]
        selected_set = self.env.selected_set + [selected]
        reward = self.env.compute_final_reward(selected_set)
        selected = np.array([x[1] for x in reward]).argsort()[-5:][::-1]

        file = open(str(self.dim) + "D_Result.txt", "w+")
        file.write(str(self.dim) + "D_Result:\n\n")
        file.write('Primary Score: ' + str(self.env.primary_reward) + '\n\n')
        if self.env.mode == 1:
            for i in selected:
                file.write('Descriptor: ' + str([self.env.trace_descriptors(int(x)) for x in reward[i][0]]) + '\n')
                file.write('Score: ' + str(reward[i][1]) + '\n\n')
        else:
            image = 1
            for i in selected:
                file.write('Descriptor: ' + str([self.env.trace_descriptors(int(x)) for x in reward[i][0]]) + '\n')
                file.write('Score: ' + str(reward[i][2]) + '    RMSE: ' + str(1/reward[i][1] - 1) + '\n\n')
                if (self.dim == 1) & draw:
                    X_train, X_test, y_train, y_test = train_test_split(self.env.features[:, reward[i][0]].reshape(-1, 1),
                                                                        self.env.target, test_size=0.3,random_state=0)
                    clf = LinearRegression().fit(X_train, y_train)
                    plt.title('Result of regression ' + str(image) + ' score: ' + str(reward[i][2]))
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.scatter(X_test, y_test, c='r')
                    plt.plot(X_test, clf.predict(X_test))
                    plt.savefig('Result of regression ' + str(image))
                    plt.show()
                    image += 1

        file.seek(0)
        print(file.read())
        file.close()

        # Plot the rewards
        plt.title('Rewards of each episode')
        plt.xlabel('Episode')
        plt.plot(list(range(len(self.score_list))), self.score_list)
        plt.show()
        self.average_sliding_window(self.score_list, 10)
        print(self.env.features.shape)
        self.output_feature_space(self.env.tag, self.env.features)


MODE = 0
ACT = ['exp', 'sum', 'p-1','p+2', 'div', 'sqr', 'sub', 'log', 'mul']
MAX_ITER = 1000
EPISODE = 5
# act = ['exp', 'sqr', 'div', 'sub', 'mul']
# act = ['exp', 'sum', 'p-1','p+2', 'div', 'sqr']
# act = ['exp', 'sum', 'p-1']
# act = ['exp', 'div', 'p+2']

eng = AFM('train.csv', 3, ACT, MODE, EPISODE) # 'completedata.csv', 2 'testingdata.csv',4
increase_dim = False
for iteration in range(MAX_ITER):
    eng.new_loop(iteration)
    eng.main()
    action = ['Y', 0, 0]
    if eng.loop_control(action):
        break
