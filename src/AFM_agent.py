import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import l1_min_c
import itertools

class Agent:

    def __init__(self, primary_features, label, features_tag, act, max_comp=3):
        self.action = act
        self.max_comp = max_comp
        self.tag = [['non', i] for i in features_tag]
        self.comp = [0 for i in features_tag]
        self.done = [False for i in features_tag]
        self.features = primary_features
        self.label = label
        self.selected_set = []
        self.accuracy = [self.compute_accuracy(primary_features[:, i]) for i in range(len(features_tag))]
        self.primary_accuracy = max(self.accuracy)
        self.index = {}
        for i in range(len(features_tag)):
            self.index['non,' + features_tag[i]] = i
        self.memory = []

    def reset(self, iter):
        return iter % 7

    def step(self, state, action):

        # exp() defined as operation 'exp'
        if action == 'exp':
            accuracy, next_state, done = self.compute_fea('exp,' + str(state),
                                                          lambda x: np.exp(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^-1 defined as operation 'p-1'
        if action == 'p-1':
            accuracy, next_state, done = self.compute_fea('p-1,' + str(state),
                                                          lambda x: np.power(x, -1).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)

        # ^2 defined as operation 'p+2'
        if action == 'p+2':
            accuracy, next_state, done = self.compute_fea('p+2,' + str(state),
                                                          lambda x: np.power(x, 2).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^6 defined as operation 'p+6'
        if action == 'p+6':
            accuracy, next_state, done = self.compute_fea('p+6,' + str(state),
                                                          lambda x: np.power(x, 6).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^sqrt defined as operation 'sqr'
        if action == 'sqr':
            accuracy, next_state, done = self.compute_fea('sqr,' + str(state),
                                                          lambda x: np.power(x, 1/2).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # ^cbrt defined as operation 'cbr'
        if action == 'cbr':
            accuracy, next_state, done = self.compute_fea('cbr,' + str(state),
                                                          lambda x: np.power(x, 1/3).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # log defined as operation 'log'
        if action == 'log':
            accuracy, next_state, done = self.compute_fea('log,' + str(state),
                                                          lambda x: np.log(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        #sin defined as operation 'sin'
        if action == 'sin':
            accuracy, next_state, done = self.compute_fea('sin,' + str(state),
                                                          lambda x: np.sin(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)
        # cos defined as operation 'cos'
        if action == 'cos':
            accuracy, next_state, done = self.compute_fea('cos,' + str(state),
                                                          lambda x: np.cos(x).reshape(self.features.shape[0], 1),
                                                          self.features[:, state], state)

        # + defined as operation 'sum'
        if action == 'sum':
            accuracy_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                accuracy, next_state, done = self.compute_fea('sum,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x + y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                accuracy_list.append(accuracy)
                next_state_list.append(next_state)
            accuracy = max(accuracy_list)
            max_index = np.argmax(accuracy_list)
            next_state = next_state_list[max_index]

        # - defined as operation 'sub'
        if action == 'sub':
            accuracy_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                accuracy, next_state, done = self.compute_fea('sub,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x - y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                accuracy_list.append(accuracy)
                next_state_list.append(next_state)
            accuracy = max(accuracy_list)
            max_index = np.argmax(accuracy_list)
            next_state = next_state_list[max_index]

        # * defined as operation 'mul'
        if action == 'mul':
            accuracy_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                accuracy, next_state, done = self.compute_fea('mul,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x * y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                accuracy_list.append(accuracy)
                next_state_list.append(next_state)
            accuracy = max(accuracy_list)
            max_index = np.argmax(accuracy_list)
            next_state = next_state_list[max_index]

        # / defined as operation 'div'
        if action == 'div':
            accuracy_list = []
            next_state_list = []
            for j in range(len(self.tag)):
                if self.done[j]: continue
                accuracy, next_state, done = self.compute_fea('div,' + str(state) + ',' + str(j),
                                                              lambda x, y: (x / y).reshape(self.features.shape[0], 1),
                                                              (self.features[:, state], self.features[:, j]), state, j)
                accuracy_list.append(accuracy)
                next_state_list.append(next_state)
            accuracy = max(accuracy_list)
            max_index = np.argmax(accuracy_list)
            next_state = next_state_list[max_index]

        # reward = accuracy - self.primary_accuracy
        reward = accuracy
        self.memory.append([state, action, reward, next_state, done])
        return reward, next_state, done

    def compute_fea(self, operator, func, variables, state, j=0):

        if self.done[state]:
            accuracy = 0
            next_state = state
            done = True
            return accuracy, next_state, done

        if operator in self.index:
            index = self.index[operator]
            next_state = index
            if self.accuracy[index] != -1:
                accuracy = self.accuracy[index]
            else:
                accuracy = self.compute_accuracy(self.features[:, index])
                self.accuracy[index] = accuracy
            done = self.done[index]
            return accuracy, next_state, done

        if operator[:3] in ['sum', 'mul']:
            new_operator = operator.split(',')
            delimiter = ','
            new_operator = delimiter.join([new_operator[0], new_operator[2], new_operator[1]])
            if new_operator in self.index:
                index = self.index[new_operator]
                self.index[operator] = index
                next_state = index
                if self.accuracy[index] != -1:
                    accuracy = self.accuracy[index]
                else:
                    accuracy = self.compute_accuracy(self.features[:, index])
                    self.accuracy[index] = accuracy
                done = self.done[index]
                return accuracy, next_state, done

        self.index[operator] = len(self.tag)
        tag = operator.split(',')
        for i in range(1, len(tag)):
            tag[i] = int(tag[i])
        self.tag.append(tag)
        if type(variables) == tuple:
            new_fea = func(variables[0], variables[1])
            self.comp.append(max(self.comp[state], self.comp[j]) + 1)
        else:
            new_fea = func(variables)
            self.comp.append(self.comp[state] + 1)
        if (np.max(np.abs(new_fea[:])) >= 1e-11) & (np.max(np.abs(new_fea[:])) <= 1e11):
            if np.max(np.abs(new_fea[:] - new_fea[0])) >= 1e-8:
                accuracy = self.compute_accuracy(new_fea)
                next_state = len(self.accuracy)
                if (self.comp[-1]) >= self.max_comp:
                    done = True
                else:
                    done = False

            else:
                accuracy = 0
                next_state = state
                done = True
        else:
            new_fea = np.zeros(new_fea.shape)
            accuracy = 0
            next_state = state
            done = True

        self.features = np.hstack((self.features, new_fea))
        self.accuracy.append(accuracy)
        self.done.append(done)

        return accuracy, next_state, done

    def compute_accuracy(self, feature, final=False):

        if final:
            accuracy = []
            comb = itertools.product(*feature)
            for i in comb:
                feature = self.features[:, i]
                X_train, X_test, y_train, y_test = train_test_split(feature, self.label, test_size=0.3,
                                                                    random_state=0)
                clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto', max_iter=1e3).fit(
                    X_train, y_train)
                accuracy.append([i, clf.score(X_test, y_test)])
            return accuracy

        if self.selected_set == []:
            X_train, X_test, y_train, y_test = train_test_split(feature.reshape(-1, 1), self.label, test_size=0.3,
                                                                random_state=0)
            clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto', max_iter = 1e3).fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
        else:
            accuracy = 0
            comb = itertools.product(*self.selected_set)
            for i in comb:
                fea = np.hstack((feature.reshape(-1, 1), self.features[:,i]))
                X_train, X_test, y_train, y_test = train_test_split(fea, self.label, test_size=0.3,
                                                                    random_state=0)
                clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto', max_iter=1e3).fit(
                    X_train, y_train)
                accuracy = max(accuracy, clf.score(X_test, y_test))
        return accuracy


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
    # def sis_score(self, fea, res):
    #     convexpair = 0
    #     overlap = 0
    #     mindist = -1e10
    #     isoverlap = False
    #     overlap_length = 0
    #     score = []
    #     for i in range(len(Global.sample_group)):
    #         featemp1 = []
    #         for k in range(Global.sample_group[i][0], Global.sample_group[i][1] + 1):
    #             if res[k] <= 1:
    #                 featemp1.append(fea[k])
    #         for j in range(i + 1, len(Global.sample_group)):
    #             featemp2 = []
    #             for k in range(Global.sample_group[j][0], Global.sample_group[j][1] + 1):
    #                 if res[k] <= 1:
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
