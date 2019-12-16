import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import l1_min_c

def identify_descriptors(lastop, fea, res):
    N = 5   #number of outputs
    fea_name = []
    # clf = linear_model.Lasso(alpha=0.3,max_iter = 1e6)

    # clf = LogisticRegression(penalty='l1', solver='saga', tol=1e-6, C = 2 ,max_iter=1e6)
    # clf.fit(fea,res)
    # clf.coef_ = clf.coef_.reshape(-1)

    X = fea
    y = res
    # X /= X.max()  # Normalize X to speed-up convergence

    # #############################################################################
    # Demo path functions

    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, 16)
    print(cs)

    print("Computing regularization path ...")
    clf = linear_model.LogisticRegression(penalty='l1', solver='saga',
                                          tol=1e-6, max_iter=int(1e6),
                                          warm_start=True)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())
    print(coefs_)
    coefs_ = np.array(coefs_)
    print(coefs_.shape)
    plt.plot(np.log10(cs), coefs_, marker='o')
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.axis('tight')
    plt.show()







    for i in range(N):
        print('lasso weights:',clf.coef_[np.argmax(clf.coef_)])
        fea_name.append(int(np.argmax(clf.coef_)))
        clf.coef_[np.argmax(clf.coef_)] = 0
    print('selected features:',fea_name)

    for i in fea_name:
        print(trace_descriptors(lastop, i))
        X_train, X_test, y_train, y_test = train_test_split(fea[:,i].reshape(-1,1), res, test_size=0.3, random_state=0)
        clf = LogisticRegression(random_state=0, solver='liblinear',multi_class='auto').fit(X_train, y_train)
        clf.predict(X_test)
        print('classification accuracy:',clf.score(X_test, y_test))




    # # testing................................
    # print('**********************')
    # print('correct operation')
    # for i in range(len(lastop)):
    #     if lastop[i] == ['div', 4, 7]:
    #         imax = i
    #         print(trace_descriptors(lastop, i))
    #         print('difference:', np.sum(np.abs(np.exp(fea[:, 0]) / np.power(fea[:, 1], 2) - fea[:, i])))
    #         X_train, X_test, y_train, y_test = train_test_split(fea[:, i].reshape(-1, 1), res, test_size=0.3,
    #                                                             random_state=0)
    #         clft = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto').fit(X_train, y_train)
    #         clft.predict(X_test)
    #         print('classification accuracy:', clft.score(X_test, y_test))
    # correlation = []
    # for i in fea_name:
    #     correlation.append(np.corrcoef((fea[:,i], fea[:, imax]))[0, 1])
    # print('correlation magnitude:', correlation)

def trace_descriptors(lastop, fea_name):
    desc = [fea_name]
    item = 0
    while(True):
        if type(desc[item]) == int:
            operation = lastop[desc[item]]
            if operation[0] == 'exp': temp = ['exp(',operation[1],')']
            elif operation[0] == 'p-1': temp = ['(',operation[1],')**-1']
            elif operation[0] == 'p+2': temp = ['(',operation[1],')**2']
            elif operation[0] == 'p+3': temp = ['(',operation[1],')**3']
            elif operation[0] == 'p+6': temp = ['(',operation[1],')**6']
            elif operation[0] == 'sqr': temp = ['(',operation[1],')**(1/2)']
            elif operation[0] == 'cbr': temp = ['(',operation[1],')**(1/3)']
            elif operation[0] == 'log': temp = ['log(',operation[1],')']
            elif operation[0] == 'sin': temp = ['sin(',operation[1],')']
            elif operation[0] == 'cos': temp = ['cos(',operation[1],')']
            elif operation[0] == 'sum': temp = ['(',operation[1],'+',operation[2],')']
            elif operation[0] == 'sub': temp = ['(',operation[1],'-',operation[2],')']
            elif operation[0] == 'mul': temp = ['(',operation[1],'*',operation[2],')']
            elif operation[0] == 'div': temp = ['(',operation[1],'/',operation[2],')']
            elif operation[0] == 'non': temp = [operation[1]]
            else:
                print('Warning: invalid operation')
                temp = []
            desc[item:item+1] = temp
        item += 1
        if item == len(desc):
            break
    temp = ''
    for item in desc:
        temp += item
    desc = temp
    return desc

# def identify_descriptors(lastop, fea_name, comp):
#     for c in range(comp):
#         desc = ''
#         startnew = 1
#         loc = 0
#         for i in range(len(fea_name)):
#             if ((fea_name[i]>='0') & (fea_name[i]<='9')):
#                 if startnew == 1:
#                     k = 0
#                     desc += fea_name[loc:i]
#                 k = k*10 + int(fea_name[i])
#                 startnew = 0
#                 loc = i+1
#             else:
#                 if startnew == 0:
#                     startnew = 1
#                     desc += lastop[k]
#         desc += fea_name[loc:]
#         fea_name = desc
#     return desc
