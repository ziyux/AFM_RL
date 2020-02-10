import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import l1_min_c
import itertools

class Agent:

    def __init__(self, primary_features, target, fea_name, act, mode, max_comp):
        self.mode = mode
        self.action = act
        self.max_comp = max_comp
        self.tag = [['non', i] for i in fea_name]
        self.comp = [0 for i in fea_name]
        self.done = [False for i in fea_name]
        self.features = primary_features
        self.target = target
        self.selected_set = []
        self.reward = [self.compute_reward(primary_features[:, i].reshape(-1, 1))
                       for i in range(len(fea_name))]
        self.primary_reward = max(self.reward + [self.compute_reward(primary_features[:,:])])
        self.index = {}
        for i in range(len(fea_name)):
            self.index['non,' + fea_name[i]] = i
        self.memory = []

    def reset(self, iter):
        return iter % 14

    def step(self, state, action):

        # exp() defined as operation 'exp'
        if action == 'exp':
            reward, next_state, done = self.compute_fea('exp,' + str(state),
                                                          lambda x: np.exp(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^-1 defined as operation 'p-1'
        if action == 'p-1':
            reward, next_state, done = self.compute_fea('p-1,' + str(state),
                                                          lambda x: np.power(x, -1).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)

        # ^2 defined as operation 'p+2'
        if action == 'p+2':
            reward, next_state, done = self.compute_fea('p+2,' + str(state),
                                                          lambda x: np.power(x, 2).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^6 defined as operation 'p+6'
        if action == 'p+6':
            reward, next_state, done = self.compute_fea('p+6,' + str(state),
                                                          lambda x: np.power(x, 6).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^sqrt defined as operation 'sqr'
        if action == 'sqr':
            reward, next_state, done = self.compute_fea('sqr,' + str(state),
                                                          lambda x: np.power(x, 1/2).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^cbrt defined as operation 'cbr'
        if action == 'cbr':
            reward, next_state, done = self.compute_fea('cbr,' + str(state),
                                                          lambda x: np.power(x, 1/3).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # log defined as operation 'log'
        if action == 'log':
            reward, next_state, done = self.compute_fea('log,' + str(state),
                                                          lambda x: np.log(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        #sin defined as operation 'sin'
        if action == 'sin':
            reward, next_state, done = self.compute_fea('sin,' + str(state),
                                                          lambda x: np.sin(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # cos defined as operation 'cos'
        if action == 'cos':
            reward, next_state, done = self.compute_fea('cos,' + str(state),
                                                          lambda x: np.cos(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)

        # + defined as operation 'sum'
        if action == 'sum':
            reward_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                reward, next_state, done = self.compute_fea('sum,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x + y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                reward_list.append(reward)
                next_state_list.append(next_state)
            reward = max(reward_list)
            max_index = np.argmax(reward_list)
            next_state = next_state_list[int(max_index)]

        # - defined as operation 'sub'
        if action == 'sub':
            reward_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                reward, next_state, done = self.compute_fea('sub,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x - y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                reward_list.append(reward)
                next_state_list.append(next_state)
            reward = max(reward_list)
            max_index = np.argmax(reward_list)
            next_state = next_state_list[int(max_index)]

        # * defined as operation 'mul'
        if action == 'mul':
            reward_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                reward, next_state, done = self.compute_fea('mul,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x * y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                reward_list.append(reward)
                next_state_list.append(next_state)
            reward = max(reward_list)
            max_index = np.argmax(reward_list)
            next_state = next_state_list[int(max_index)]

        # / defined as operation 'div'
        if action == 'div':
            reward_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                reward, next_state, done = self.compute_fea('div,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x / y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                reward_list.append(reward)
                next_state_list.append(next_state)
            reward = max(reward_list)
            max_index = np.argmax(reward_list)
            next_state = next_state_list[int(max_index)]

        self.memory.append([state, action, reward, next_state, done])
        return reward, next_state, done

    def compute_fea(self, operator, func, variables, state, j=0):

        if self.done[state]:
            reward = 0
            next_state = state
            done = True

            if np.max(np.abs(self.features[:,state])) == 0:
                print(self.features[:,state])
            return reward, next_state, done

        if operator in self.index:
            index = self.index[operator]
            if index == -1:
                reward = 0
                next_state = state
                if type(variables) == tuple:
                    comp = max(self.comp[state], self.comp[j]) + 1
                else:
                    comp = self.comp[state] + 1

                if comp >= self.max_comp:
                    done = True
                else:
                    done = False
            else:
                next_state = index
                if self.reward[index] != -1:
                    reward = self.reward[index]
                else:
                    reward = self.compute_reward(self.features[:, index])
                    self.reward[index] = reward
                done = self.done[index]
            return reward, next_state, done

        if operator[:3] in ['sum', 'mul']:
            new_operator = operator.split(',')
            delimiter = ','
            new_operator = delimiter.join([new_operator[0], new_operator[2], new_operator[1]])
            if new_operator in self.index:
                index = self.index[new_operator]
                self.index[operator] = index
                next_state = index
                if self.reward[index] != -1:
                    reward = self.reward[index]
                else:
                    reward = self.compute_reward(self.features[:, index])
                    self.reward[index] = reward
                done = self.done[index]
                return reward, next_state, done

        ##########
        # print(operator)
        # print('variables', variables)
        if np.max(np.abs(variables[0]))==0:
            print(variables)
        ##########

        if type(variables) == tuple:
            new_fea = func(variables[0], variables[1])
            comp = max(self.comp[state], self.comp[j]) + 1
        else:
            new_fea = func(variables)
            comp = self.comp[state] + 1

        if comp >= self.max_comp:
            done = True
        else:
            done = False
        ###########
        # print('newfea', new_fea[:])
        ###########
        if (np.max(np.abs(new_fea[:])) >= 1e-11) & (np.max(np.abs(new_fea[:])) <= 1e11):
            if np.max(np.abs(new_fea[:] - new_fea[0])) >= 1e-8:
                reward = self.compute_reward(new_fea.reshape(-1, 1))
                next_state = len(self.reward)

                self.index[operator] = len(self.tag)
                tag = operator.split(',')
                for i in range(1, len(tag)):
                    tag[i] = int(tag[i])
                self.tag.append(tag)
                self.features = np.hstack((self.features, new_fea))
                self.reward.append(reward)
                self.comp.append(comp)
                self.done.append(done)

            else:
                reward = 0
                next_state = state
                self.index[operator] = -1

        else:
            # new_fea = np.zeros(new_fea.shape)
            reward = 0
            next_state = state
            self.index[operator] = -1

        return reward, next_state, done

    def compute_reward(self, feature):
        if self.selected_set == []:
            X_train, X_test, y_train, y_test = train_test_split(feature, self.target, test_size=0.3,
                                                                random_state=0)
            if self.mode == 1:
                clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto', max_iter = 1e3)\
                    .fit(X_train, y_train)
                reward = clf.score(X_test, y_test)
            else:
                clf = LinearRegression().fit(X_train, y_train)
                # rmse = np.sqrt(mean_squared_error(clf.predict(X_test), y_test))
                # reward = 1/(1+rmse)
                reward = clf.score(X_test, y_test)

        else:
            reward = 0
            comb = itertools.product(*self.selected_set)
            if self.mode == 1:
                for i in comb:
                    fea = np.hstack((feature.reshape(-1, 1), self.features[:,i]))
                    X_train, X_test, y_train, y_test = train_test_split(fea, self.target, test_size=0.3,
                                                                        random_state=0)
                    clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto', max_iter=1e3)\
                        .fit(X_train, y_train)
                    reward = max(reward, clf.score(X_test, y_test))
            else:
                for i in comb:
                    fea = np.hstack((feature.reshape(-1, 1), self.features[:,i]))
                    X_train, X_test, y_train, y_test = train_test_split(fea, self.target, test_size=0.3,
                                                                        random_state=0)
                    clf = LinearRegression().fit(X_train, y_train)
                    # rmse = np.sqrt(mean_squared_error(clf.predict(X_test), y_test))
                    # reward = max(reward, 1 / (1 + rmse))
                    reward = max(reward, clf.score(X_test, y_test))
        return reward

    def compute_final_reward(self, feature):

        reward = []
        comb = itertools.product(*feature)
        if self.mode == 1:
            for i in comb:
                feature = self.features[:, i]
                X_train, X_test, y_train, y_test = train_test_split(feature, self.target, test_size=0.3,
                                                                    random_state=0)
                clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto', max_iter=1e3)\
                    .fit(X_train, y_train)
                reward.append([i, clf.score(X_test, y_test)])
        else:
            for i in comb:
                feature = self.features[:, i]
                X_train, X_test, y_train, y_test = train_test_split(feature, self.target, test_size=0.3,
                                                                    random_state=0)
                clf = LinearRegression().fit(X_train, y_train)
                rmse = np.sqrt(mean_squared_error(clf.predict(X_test), y_test))
                reward.append([i, 1 / (1 + rmse), clf.score(X_test, y_test)])
        return reward

    def trace_descriptors(self, fea_name):
        desc = [fea_name]
        item = 0
        while (True):
            if type(desc[item]) == int:
                operation = self.tag[desc[item]]
                if operation[0] == 'exp':
                    temp = ['exp(', operation[1], ')']
                elif operation[0] == 'p-1':
                    temp = ['(', operation[1], ')^-1']
                elif operation[0] == 'p+2':
                    temp = ['(', operation[1], ')^2']
                elif operation[0] == 'p+3':
                    temp = ['(', operation[1], ')^3']
                elif operation[0] == 'p+6':
                    temp = ['(', operation[1], ')^6']
                elif operation[0] == 'sqr':
                    temp = ['(', operation[1], ')^(1/2)']
                elif operation[0] == 'cbr':
                    temp = ['(', operation[1], ')^(1/3)']
                elif operation[0] == 'log':
                    temp = ['log(', operation[1], ')']
                elif operation[0] == 'sin':
                    temp = ['sin(', operation[1], ')']
                elif operation[0] == 'cos':
                    temp = ['cos(', operation[1], ')']
                elif operation[0] == 'sum':
                    temp = ['(', operation[1], '+', operation[2], ')']
                elif operation[0] == 'sub':
                    temp = ['(', operation[1], '-', operation[2], ')']
                elif operation[0] == 'mul':
                    temp = ['(', operation[1], '*', operation[2], ')']
                elif operation[0] == 'div':
                    temp = ['(', operation[1], '/', operation[2], ')']
                elif operation[0] == 'non':
                    temp = [operation[1]]
                else:
                    print('Warning: invalid operation')
                    temp = []
                desc[item:item + 1] = temp
            item += 1
            if item == len(desc):
                break
        temp = ''
        for item in desc:
            temp += item
        desc = temp
        return desc

    # def sis_score(self, fea, target):
    #     convexpair = 0
    #     overlap = 0
    #     mindist = -1e10
    #     isoverlap = False
    #     overlap_length = 0
    #     score = []
    #     for i in range(len(Global.sample_group)):
    #         featemp1 = []
    #         for k in range(Global.sample_group[i][0], Global.sample_group[i][1] + 1):
    #             if target[k] <= 1:
    #                 featemp1.append(fea[k])
    #         for j in range(i + 1, len(Global.sample_group)):
    #             featemp2 = []
    #             for k in range(Global.sample_group[j][0], Global.sample_group[j][1] + 1):
    #                 if target[k] <= 1:
    #                     featemp2.append(fea[k])
    #             num, length = convex1d_overlap(featemp1, featemp2)
    #             overlap += num
    #             convexpair += 1
    #             minlen = min(np.max(featemp1) - np.min(featemp1), np.max(featemp2) - np.min(featemp2))
    #             if (length >= 0): isoverlap = True
    #             if (length < 0):
    #                 if length > mindist: mindist = length
    #             elif (length >= 0) & (minlen == 0):
    #                 overlap_length += 1
    #             elif (length >= 0) & (minlen >= 0):
    #                 overlap_length += length / minlen
    #     score.append(1.0 / (1.0 + float(overlap)))
    #     if isoverlap:
    #         score.append(1.0 / (1.0 + length / convexpair))
    #     else:
    #         score.append(-mindist)
    #     return score
    #
    # def convex1d_overlap(self, set1, set2):
    #     num = 0
    #     mini = np.min(set2) - Global.WIDTH
    #     maxi = np.max(set2) + Global.WIDTH
    #     for i in range(len(set1)):
    #         if (set1[i] > mini) & (set1[i] < maxi): num += 1
    #     mini = np.min(set1) - Global.WIDTH
    #     maxi = np.max(set1) + Global.WIDTH
    #     for i in range(len(set2)):
    #         if (set2[i] > mini) & (set2[i] < maxi): num += 1
    #     length = min(np.max(set1), np.max(set2)) - max(np.min(set1), np.min(set2))
    #     return num, length
