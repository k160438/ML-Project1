import os
import sys
import model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

rootpath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(rootpath)

if __name__ == "__main__":
    X_train = np.load('data/X_train_hog.npy')
    X_test = np.load('data/X_test_hog.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    print("Loading data!")
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    
    # model = model.LogisticRegression(X_train.shape[1])
    # model.setOptimizer('Langevin', 0.1, 500, 1)
    
    model = model.LDA(X_train.shape[1], 30)
    
    model.fit(X_train, y_train)
    print("Test on test data:")
    # test_loss = model.lossFunction(X_test, y_test)
    test_acc = model.evaluation(X_test, y_test)
    print("Test acc: {}".format(test_acc))

    # print("loss: {} \nacc: {}".format(test_loss, test_acc))
    # print('svm classification')
    # svc = SVC(gamma='auto')
    # c_range = [1]
    # # gamma_range = np.logspace(-5, 3, 5, base=2)
    # param_grid = [{'kernel': ['linear'], 'C': c_range}]
    # grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=4)
    # clf = grid.fit(X_train, y_train)
    # acc = grid.score(X_test, y_test)
    # print("acc is {}".format(acc))
