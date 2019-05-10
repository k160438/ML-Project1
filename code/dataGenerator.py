import os
import sys
import cv2
import numpy as np

rootPath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(rootPath)
np.random.seed(101)

# the pixels to padding on each border
padding = 500


def PostiveSamples():
    X_train = []
    X_test = []
    print("Generating positive samples...")
    for i in range(1, 11):
        filename = 'data/FDDB-folds/FDDB-fold-{}-ellipseList.txt'.format(
            str(i).zfill(2))
        images_l = open(filename)
        imgName = images_l.readline().strip()
        while imgName != '':
            img = cv2.imread("data/{}.jpg".format(imgName))
            img = cv2.copyMakeBorder(
                img, padding, padding, padding, padding, borderType=cv2.BORDER_REPLICATE)
            try:
                numOfFaces = int(images_l.readline().strip())
            except:
                print('a')
                print(type(imgName))
                # print(imgName, images_l.readlines())
                exit()

            for j in range(numOfFaces):
                major_radius, minor_radius, angle, center_x, center_y, _ = images_l.readline().split()
                # print(major_radius, minor_radius, angle, center_x, center_y)
                major_radius, minor_radius, angle, center_x, center_y = float(major_radius), float(
                    minor_radius), float(angle), float(center_x)+padding, float(center_y)+padding
                width = minor_radius*8/3
                height = major_radius*8/3
                # if center_y<height/2 or center_x < width/2 or center_y+height/2 > img.shape[0] or center_x + width/2 > img.shape[1]:
                #     print("Outside of image!")
                #     print(major_radius, minor_radius)
                face = img[int(center_y-height/2):int(center_y+height/2),
                           int(center_x - width/2):int(center_x + width/2)]
                try:
                    face = cv2.resize(face, (96, 96))
                except:
                    continue
                if i < 9:
                    X_train.append(face)
                else:
                    X_test.append(face)
                # print(face.shape)
                # cv2.imshow('face', face)
                # print(width, height)
                # cv2.rectangle(img, (int(center_x - width/2), int(center_y-height/2)),
                #               (int(center_x+width/2), int(center_y+height/2)), (0, 0, 255))
            # cv2.imshow("visualized_positive_img", img)
            # cv2.waitKey(0)
            # cv2.imwrite("data/visual/positive_{}.jpg".format(imgName.replace('/', '_')), img)
            imgName = images_l.readline().strip()
        print("{} loading finish!".format(filename))

    return np.array(X_train), np.array(X_test)


def NegativeSamples():
    X_train = []
    X_test = []
    print("Generating negative samples...")
    for i in range(1, 11):
        if i > 4 and i < 10:
            continue
        filename = 'data/FDDB-folds/FDDB-fold-{}-ellipseList.txt'.format(
            str(i).zfill(2))
        images_l = open(filename)
        imgName = images_l.readline().strip()
        while imgName != '':
            img = cv2.imread("data/{}.jpg".format(imgName))
            img = cv2.copyMakeBorder(
                img, padding, padding, padding, padding, borderType=cv2.BORDER_REPLICATE)
            try:
                numOfFaces = int(images_l.readline().strip())
            except:
                print('a')
                print(type(imgName))
                # print(imgName, images_l.readlines())
                exit()

            for j in range(numOfFaces):
                major_radius, minor_radius, angle, center_x, center_y, _ = images_l.readline().split()
                # print(major_radius, minor_radius, angle, center_x, center_y)
                major_radius, minor_radius, angle, center_x, center_y = float(major_radius), float(
                    minor_radius), float(angle), float(center_x)+padding, float(center_y)+padding
                width = minor_radius
                height = major_radius
                delta_a = height*2/3
                delta_b = width*2/3
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if a == 0 and b == 0:
                            continue
                        n_face = img[int(center_y-height+a*delta_a):int(center_y+height+a*delta_a),
                                     int(center_x - width+b*delta_b):int(center_x + width+b*delta_b)]
                        try:
                            n_face = cv2.resize(n_face, (96, 96))
                        except:
                            continue
                        # cv2.rectangle(img, (int(center_x - width+b*delta_b), int(center_y-height+a*delta_a)),
                        #               (int(center_x+width+b*delta_b), int(center_y+height+a*delta_a)), (0, 0, 0))
                        if i < 5:
                            X_train.append(n_face)
                        else:
                            X_test.append(n_face)
                        # print(n_face.shape)
                        # cv2.imshow('n_face_{}'.format(j), n_face)

            # cv2.imshow("visualized_positive_img", img)
            # cv2.waitKey(0)
            # cv2.imwrite("data/visual/negative_{}.jpg".format(imgName.replace('/', '_')), img)
            imgName = images_l.readline().strip()
        print("{} loading finish!".format(filename))

    return np.array(X_train), np.array(X_test)


def GetDataset():
    X_train_p, X_test_p = PostiveSamples()
    X_train_n, X_test_n = NegativeSamples()
    print("Generate positive samples:\nTraining: {}\nTest: {}".format(
        X_train_p.shape, X_test_p.shape))
    print("Generate negative samples:\nTraining: {}\nTest: {}".format(
        X_train_n.shape, X_test_n.shape))
    y_train_p = np.ones(len(X_train_p))
    y_test_p = np.ones(len(X_test_p))
    y_train_n = - np.ones(len(X_train_n))
    y_test_n = -np.ones(len(X_test_n))
    X_train = np.concatenate((X_train_p, X_train_n), axis=0)
    X_test = np.concatenate((X_test_p, X_test_n), axis=0)
    y_train = np.concatenate((y_train_p, y_train_n), axis=0)
    y_test = np.concatenate((y_test_p, y_test_n), axis=0)
    indices_train = np.arange(len(X_train))
    indices_test = np.arange(len(X_test))
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)
    return X_train[indices_train], X_test[indices_test], y_train[indices_train], y_test[indices_test]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = GetDataset()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    np.save('data/X_train', X_train)
    np.save('data/X_test', X_test)
    np.save('data/y_train', y_train)
    np.save('data/y_test', y_test)