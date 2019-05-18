import os
import sys
from skimage.feature import hog
import numpy as np
import cv2

rootpath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(rootpath)


if __name__ == "__main__":
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    X_train_hog = []
    X_test_hog = []
    print("Extract HOG features of training samples...")
    for i in range(X_train.shape[0]):
        if (i+1) % 1000 == 0:
            print("{}/{}".format(i+1, X_train.shape[0]))
        feature, hog_img = hog(
            X_train[i], 9, (16, 16), (2, 2), visualise=True, feature_vector=True)
        X_train_hog.append(feature)
    X_train_hog = np.array(X_train_hog)
    print("Finish extracting training samples!")
    print(X_train_hog.shape)
    np.save('data/X_train_hog', X_train_hog)

    print("Extract HOG features of test samples...")
    for i in range(X_test.shape[0]):
        if (i+1) % 1000 == 0:
            print("{}/{}".format(i+1, X_test.shape[0]))
        feature, hog_img = hog(
            X_test[i], 9, (16, 16), (2, 2), visualise=True, feature_vector=True)
        X_test_hog.append(feature)
    X_test_hog = np.array(X_test_hog)
    print("Finish extracting test samples!")
    print(X_test_hog.shape)
    np.save('data/X_test_hog', X_test_hog)

    # i = 1
    # feature, hog_img = hog(X_train[i], 9, (16, 16), (2, 2), visualise=True, feature_vector=True)
    # hog_img = hog_img.reshape(96,96,1)
    # hog_img = cv2.cvtColor(hog_img, cv2.COLOR_GRAY2BGR)
    # print(hog_img)
    # print(X_train[i].shape, hog_img.shape)

    # img = np.concatenate((X_train[i], hog_img))
    # cv2.imshow('hog', img)
    # cv2.waitKey(0)
