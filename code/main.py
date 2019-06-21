import os
import sys
import models
import numpy as np
from joblib import dump, load

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
    
    # print('Logistic regression')
    # model = models.LogisticRegression(X_train.shape[1])
    # model.setOptimizer('Langevin', 0.08, 7000, 1)
    # # model.setOptimizer('SGD', 0.001, 1500)
    # model.fit(X_train, y_train)
    # print("Test on test data:")
    # test_loss = model.lossFunction(X_test, y_test)
    # test_acc = model.evaluation(X_test, y_test)
    # print("loss: {} \nacc: {}".format(test_loss, test_acc))
    # model.save('data/logistic_v3.pkl')


    # print('LDA classification')
    # model = models.LDA(X_train.shape[1], 30, 0)
    # model.fit(X_train, y_train)
    # print("Test on test data:")
    # test_acc = model.evaluation(X_test, y_test)
    # print("Test acc: {}".format(test_acc))


    print('svm classification')
    kernel = 'rbf'
    model = models.SVM(kernel)
    best_model = model.fit(X_train, y_train)
    acc = model.evaluation(X_test, y_test)
    print(best_model.support_vectors_.shape)
    print(best_model.support_.shape)
    print("acc is {}".format(acc))
    print('save model...')
    dump(best_model, 'data/{}_SVM_v2.joblib'.format(kernel))

    # svm = load('data/rbf_SVM.joblib')
    # preds = svm.predict(X_test[:100])
    # print(preds)