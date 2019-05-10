import os
import sys
import cv2
import numpy as np

rootPath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(rootPath)
np.random.seed(101)

padding = 500

def PostiveImages():
    X_train = []
    X_test = []

    for i in range(1, 11):
        filename = 'data/FDDB-folds/FDDB-fold-{}-ellipseList.txt'.format(str(i).zfill(2))
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
                width = minor_radius*np.abs(np.sin(angle))*8/3
                height = major_radius*np.abs(np.sin(angle))*8/3
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


if __name__ == "__main__":
    X_train, X_test = PostiveImages()
    print(X_train.shape)
    print(X_test.shape)