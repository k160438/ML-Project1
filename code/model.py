import numpy as np
import os
import sys
from sklearn.svm import SVC

rootpath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(rootpath)

np.random.seed(101)


class LogisticRegression:
    def __init__(self, input_shape):
        self.dims = input_shape
        self.W = 0.2*(np.random.random(input_shape) - 0.5)
        self.optimizer = "SGD"
        self.learning_rate = 0.001
        self.epoch = 1000
        self.batchsize = 1
        self.sigma = 1
        self.delta_t = 0.2

    def setOptimizer(self, optimizer='SGD', learning_rate=0.001, epoch=1000, batchsize=1, sigma=1, delta_t=0.2):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batchsize = batchsize
        self.sigma = sigma
        self.delta_t = delta_t

    def lossFunction(self, X_train, y_train):
        res = 0
        for i in range(X_train.shape[0]):
            res += np.log(1 + np.exp(-np.dot(X_train[i], self.W.T)*y_train[i]))
        res /= X_train.shape[0]
        return 1000*res

    def fit(self, X_train, y_train):
        self.W = 2*(np.random.random(self.dims) - 0.5)
        print('Training start:\nMethod:{}'.format(self.optimizer.lower()))
        if self.optimizer.lower() == 'minibatch':
            for e in range(self.epoch):
                batch = np.random.choice(
                    range(X_train.shape[0]), self.batchsize, replace=False)
                grad = np.zeros(self.dims)
                for i in batch:
                    grad_p = y_train[i]/(1+np.exp(y_train[i]
                                                  * np.dot(X_train[i], self.W.T)))
                    grad += grad_p*X_train[i]
                grad /= self.batchsize
                self.W += grad*self.learning_rate
                if (e+1) % 10 == 0:
                    # print(grad[:20])
                    # print(self.W[:10])
                    acc = self.evaluation(X_train, y_train)
                    loss = self.lossFunction(X_train, y_train)
                    print("epoch {}/{}: training loss is {}, acc is {}".format(e+1,
                                                                               self.epoch, loss, acc))
        elif self.optimizer.lower() == 'sgd':
            for e in range(self.epoch):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                for i in indices:
                    self.W += self.learning_rate * \
                        y_train[i]/(1+np.exp(y_train[i] *
                                             np.dot(X_train[i], self.W.T))) * X_train[i]
                acc = self.evaluation(X_train, y_train)
                loss = self.lossFunction(X_train, y_train)
                print(
                    "epoch {}/{}: training loss is {}, acc is {}".format(e+1, self.epoch, loss, acc))

        elif self.optimizer.lower() == 'langevin':
            for e in range(self.epoch):
                grad_p = y_train / \
                    (1 + np.exp(y_train * np.dot(X_train, self.W)))
                # print(grad_p.shape)
                grad = np.mean((X_train.T * grad_p.T).T, axis=0)
                # print(grad.shape)
                # self.W += grad*self.learning_rate

                # the Gaussian noise, scale is variance
                epsilon = np.random.normal(scale=1, size=self.dims)
                self.W += self.learning_rate * (grad + self.delta_t * epsilon)
                acc = self.evaluation(X_train, y_train)
                loss = self.lossFunction(X_train, y_train)
                print(
                    "epoch {}/{}: training loss is {}, acc is {}".format(e+1, self.epoch, loss, acc))
        else:
            print("No suitable optimizer")

    def evaluation(self, X_test, y_test):
        count_true = 0
        for i in range(X_test.shape[0]):
            if np.dot(X_test[i], self.W.T) > 0:
                pred = 1
            else:
                pred = -1
            if pred == y_test[i]:
                count_true += 1
        return count_true / X_test.shape[0]


class LDA:
    def __init__(self, Input_shape, bins=50):
        self.dims = Input_shape
        self.W = np.zeros(self.dims)
        self.threshold = 0
        self.delta_bins = bins
        self.center_p = None
        self.center_n = None
        self.sigma_sqaure_p = None
        self.sigma_sqaure_n = None

    def fit(self, X_train, y_train):
        X_p = []
        X_n = []
        for i in range(X_train.shape[0]):
            if y_train[i] == 1:
                X_p.append(X_train[i])
            else:
                X_n.append(X_train[i])
        X_n = np.array(X_n)
        X_p = np.array(X_p)

        mean_p = np.mean(X_p, axis=0)
        mean_n = np.mean(X_n, axis=0)
        # var_p = np.cov(X_p.T)
        # var_n = np.cov(X_n.T)
        X_p = X_p - mean_p
        X_n = X_n - mean_n
        var_p = np.dot(X_p.T, X_p) / X_p.shape[0]
        var_n = np.dot(X_n.T, X_n) / X_n.shape[0]

        Sw = var_p*X_p.shape[0] + var_n*X_n.shape[0]
        Sw_inv = np.linalg.inv(Sw)
        self.W = np.dot(Sw_inv, mean_p - mean_n)
        intra_class_var = X_p.shape[0] * np.dot(self.W.T, np.dot(
            var_p, self.W)) + X_n.shape[0] * np.dot(self.W.T, np.dot(var_n, self.W))
        inter_class_var = np.square(np.dot(self.W.T, mean_p - mean_n))
        print("Intra-class variance: {}".format(intra_class_var))
        print("Inter-class variance: {}".format(inter_class_var))

        self.center_p = np.dot(self.W.T, mean_p)
        self.center_n = np.dot(self.W.T, mean_n)
        self.sigma_sqaure_p = np.dot(np.dot(self.W.T, var_p), self.W)
        self.sigma_sqaure_n = np.dot(np.dot(self.W.T, var_n), self.W)
        if self.center_p < self.center_n:
            self.W = -self.W
            self.center_p = -1 * self.center_p
            self.center_n = -1 * self.center_n
        print("Positive center: {}\nNegative center: {}".format(
            self.center_p, self.center_n))

        # delta = (self.center_p - self.center_n) / self.delta_bins

        # best_threshold = 0
        # best_acc = 0
        # for i in range(1, self.delta_bins):
        #     self.threshold = self.center_n + i * delta
        #     acc = self.evaluation(X_train, y_train)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_threshold = self.threshold
        # self.threshold = best_threshold

        # print("Best threshold: {}\nTraining acc: {}".format(
        #     best_threshold, best_acc))
        print("training acc: {}".format(self.evaluation(X_train, y_train)))


    def evaluation(self, X_test, y_test):
        count_true = 0
        for i in range(X_test.shape[0]):
            projection = np.dot(self.W.T, X_test[i])
            a = 1 / np.sqrt(2*np.pi*self.sigma_sqaure_p) * \
                np.exp(-(projection-self.center_p)
                    ** 2 / 2 / self.sigma_sqaure_p)
            b = 1 / np.sqrt(2*np.pi*self.sigma_sqaure_n) * \
                np.exp(-(projection-self.center_n)
                    ** 2 / 2 / self.sigma_sqaure_n)
            # if projection > self.threshold:
            if a > b:
                pred = 1
            else:
                pred = -1
            if pred == y_test[i]:
                count_true += 1
        return count_true / X_test.shape[0]
