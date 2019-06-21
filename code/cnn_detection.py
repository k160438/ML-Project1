import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import cv2
from sklearn.cluster import MeanShift

root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_path)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*8*8, 600)
        self.fc2 = nn.Linear(600, 150)
        self.fc3 = nn.Linear(150, 18)
        self.fc4 = nn.Linear(18, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()
net = net.to(device)
net.load_state_dict(torch.load('data/cnn_model.pkl'))
# print(net)

# coordinates of results after padding (ax, ay, bx, by)
res = []

def inFaces(res, x, y):
    for ax, ay, bx, by in res:
        if (x>ax and x<bx) and (y>ay and y<by):
            return True
    return False

# inputs: n*96*96*3
def evaluation(inputs):
    X = torch.from_numpy(inputs / 255.)
    X = torch.transpose(torch.transpose(X, 1, 3), 2, 3)

    with torch.no_grad():
        X = X.to(device)
        outputs = net(X.float())
        _, predicted = torch.max(outputs.data, 1)
    return predicted


# some examples for mdoel eval
# X_test = np.load('data/X_test.npy')
# y_test = np.load("data/y_test.npy")
# pred = evaluation(X_test[:10]).cpu().numpy()
# print(pred)
# print(y_test[:10])
# exit(0)


# face detection 
# filename: 2002_08_18_big_img_181
padding = 500
filename = 'data/2002/09/10/big/img_8075.jpg'
img = cv2.imread(filename)
a, b, _ = img.shape
img = cv2.copyMakeBorder(
                img, padding, padding, padding, padding, borderType=cv2.BORDER_REPLICATE)
# cv2.imwrite('data/visual/test.jpg', img)

stride_x = int(b / 40)
stride_y = int(a / 40)
height_start = int(a / 8)
height_end = int(a * 0.8)
height_stride = int((height_end - height_start) / 20)

# search from big to small
for height in range(height_end, height_start, -height_stride):
    if height >98:
        continue
    count = 0
    width = int(height / 3 * 2)
    sub_imgs = []
    coordinates = []
    for x in range(0, b, stride_x):
        for y in range(0, a, stride_y):
            center_x = x + padding
            center_y = y + padding
            if inFaces(res, center_x, center_y):
                continue
            sub_img = img[int(center_y-height/2):int(center_y+height/2),
                           int(center_x - width/2):int(center_x + width/2)]
            try:
                sub_img = cv2.resize(sub_img, (96, 96))
            except:
                continue
            sub_imgs.append(sub_img)
            coordinates.append((center_x, center_y))
    sub_imgs = np.array(sub_imgs)
    predicted = evaluation(sub_imgs).cpu().numpy()

    face_coordinates = []
    for i in range(predicted.shape[0]):
        if predicted[i]:
            face_coordinates.append(coordinates[i])
            cv2.rectangle(img, (int(coordinates[i][0] - width/2), int(coordinates[i][1]-height/2)),
                              (int(coordinates[i][0]+width/2), int(coordinates[i][1]+height/2)), (0, 0, 255))
    print(len(predicted))
    count += np.sum(predicted)
    print('height = {}, face = {}'.format(height, count))
    face_coordinates = np.array(face_coordinates)
    clustering = MeanShift(bandwidth=50).fit(face_coordinates)
    print(len(clustering.cluster_centers_))
    print(len(clustering.labels_))
    i = 0
    for x, y in clustering.cluster_centers_:
        count = 0
        count_num = 0
        for j in range(len(face_coordinates)):
            if clustering.labels_[j] == i:
                count_num += 1
                count += (face_coordinates[i][0] - x)**2+(face_coordinates[i][1]-y)**2
        print("center:({}, {})\tNum:{} Deviate:{}".format(x, y, count_num, count / count_num))
        i+=1
        cv2.circle(img, (int(x), int(y)), radius=count_num, color=(255, 0, 0))
    cv2.imwrite('data/visual/face_cluster_h={}.jpg'.format(height), img)
    break
    # 通过聚类，可以再多个尺度下判断是否是人脸，推荐从大到小，有很多脸属于同一个