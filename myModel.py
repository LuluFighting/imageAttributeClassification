from  torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import glob
import cv2
import imageProcess
import random
import numpy as np
import torch.utils.data as Data
DATA_PATH = './images'
LR = 0.01
BATCH_SIZE = 64
EPOCH = 2
resnet_model = models.resnet50(pretrained=True)
fc_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(fc_features,2)
''''''
def create_image_lists(image_dir):
    my_dir = Path(image_dir)
    if not my_dir.exists():
        print("Image directory '" + image_dir + "' not found.")
        return None
    sub_dirs = [x[0] for x in os.walk(image_dir)]  #获取当前目录下的所有目录
    image_list,label_list = [],[]
    imageProcesser = imageProcess.Img_process()
    is_root_dir = True        #不要在根目录下放图像
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg','jpeg','JPG','JPEG']
        file_list = [] #文件名集合
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir,dir_name,'*.'+ extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print("No files found")
            continue
        for file_name in file_list:
            portion = os.path.splitext(file_name)
            label_name = portion[0] + '.txt'
            filePtr = open(label_name,'r')
            label = int(filePtr.read(1))
            ori_image = cv2.imread(file_name)
            random_angle = random.randint(0,90)
            image_rotate = imageProcesser.rotation(ori_image,(ori_image.shape[1]/2,ori_image.shape[0]/2),random_angle,1)
            image_affine_transform = imageProcesser.affine_transform(ori_image)
            image_perspective_transform = imageProcesser.perspective_transform(ori_image)
            image_list.append(ori_image)
            image_list.append(image_rotate)
            image_list.append(image_affine_transform)
            image_list.append(image_perspective_transform)
            for i in range(4):
                label_list.append(label)
            filePtr.close()

    return image_list,label_list
def train_test_split(X,y,test_size = 0.2):
    X_num = X.shape[0]
    train_index,test_index = list(range(X_num)),[]
    test_num = int(X_num*test_size)
    for i in range(test_num):
        randomIndex = random.randint(0,len(train_index)-1)
        test_index.append(randomIndex)
        del train_index[randomIndex]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    return X_train,X_test,y_train,y_test
def main():
    image_list,label_list = create_image_lists(DATA_PATH)
    X_image = np.array(image_list)
    y_label = np.array(label_list)
    X_image = np.transpose(X_image,[0,3,1,2])
    X_image = torch.from_numpy(X_image).type(torch.int8)
    y_label = torch.from_numpy(y_label).type(torch.int8)
    X_train,X_test,y_train,y_test = train_test_split(X_image,y_label,test_size=0.2)
    train_dataset = Data.TensorDataset(X_train,y_train)
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    for param in resnet_model.parameters():
        param.requires_grad = False
    for param in resnet_model.fc.parameters():
        param.requires_grad = True
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,resnet_model.parameters()),lr=LR,betas=(0.9,0.99))
    for epoch in range(EPOCH):
        for step,(batch_x,batch_y) in enumerate(train_loader):
            output = resnet_model(batch_x)
            loss = loss_func(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                test_out = resnet_model(X_test)
                test_prediction = torch.max(test_out,1)[1]
                pred_y = test_prediction.data.numpy().squeeze()
                target_y = y_test.data.numpy()
                accuracy = sum(pred_y == target_y)/y_test.shape[0]
                print("Epoch: ", epoch, '| train loss:%.4f' % loss.data.numpy(), '|test accuracy : %.2f' % accuracy)
if __name__ == '__main__':
    main()

