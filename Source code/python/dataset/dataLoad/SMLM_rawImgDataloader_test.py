from torchvision import transforms
from torch.utils.data import Dataset
from imageUtils import normalize_im,normalize_maxmin
from random import choice
import matplotlib.pyplot as plt

# class train_loader(Dataset):
#     def __init__(self,X_train_norm,Y_train):
#         self.X_train_norm = X_train_norm
#         self.Y_train = Y_train
#
#     def __getitem__(self, index):
#         x_train_norm, y_train = self.X_train_norm[index],self.Y_train[index]  # 输入图像路径，标签图像路径
#         return x_train_norm, y_train      # 输入图像，标签图像（转换为tensor）
#
#     def __len__(self):
#         return len(self.X_train_norm)                       # 数据集的组数（1输入图像+1标签图像=1组）
#
# class test_loader(Dataset):
#     def __init__(self,X_test_norm,Y_test):
#         self.X_test_norm = X_test_norm
#         self.Y_test = Y_test
#
#     def __getitem__(self, index):
#         x_test_norm, y_test = self.X_test_norm[index],self.Y_test[index]  # 输入图像路径，标签图像路径
#         return x_test_norm, y_test                          # 输入图像，标签图像（转换为tensor）
#
#     def __len__(self):
#         return len(self.X_test_norm)                       # 数据集的组数（1输入图像+1标签图像=1组）

import os
import cv2
import numpy as np
import torch
class myDataLoaderTest(Dataset):
    def __init__(self,path,isTrain=True,ratio=0.95,dataNums = 1000,IsallImgUsed = False,isrotate = True, method = "ours"):
        self.path = path
        self.method = method
        self.RawImgPath     = os.path.join(self.path,'rawImg')
        self.RawImgUpPath   = os.path.join(self.path,'rawImgUp')
        # self.HeatmapImgPath = os.path.join(self.path,'HeatmapImg')

        self.isrotate = isrotate

        self.fileNames = os.listdir(self.RawImgPath)
        self.fileNames.sort(key=lambda x:int(x[:-4]))
        self.fileNames = self.fileNames[:dataNums]

        if not IsallImgUsed:          # 如果所有的图像都将被使用，则直接下一步，没必要分为训练集和测试集
            if isTrain:
                self.fileNames = self.fileNames[:int(len(self.fileNames)* ratio)]
            else:
                self.fileNames = self.fileNames[int(len(self.fileNames) * ratio):]

    def __getitem__(self, index):
        rawImgFileName      = os.path.join(self.RawImgPath      ,self.fileNames[index])  # 原始图像路径
        rawImgUpFileName    = os.path.join(self.RawImgUpPath    ,self.fileNames[index])  # 原始上采样图像路径
        # heatMapImgFileName  = os.path.join(self.HeatmapImgPath  ,self.fileNames[index])  # heatmap路径

        # 读取图像
        if self.method == "ours":
            rawImg      = cv2.imread(rawImgFileName     , cv2.IMREAD_UNCHANGED)
            rawImg = rawImg.astype(np.float32)
            rawImg = normalize_maxmin(rawImg)
            # 转换为pytorch dataloader使用的格式
            rawImg = torch.tensor(rawImg).float()
            rawImg = rawImg.view(-1, rawImg.shape[0], rawImg.shape[1])

            if self.isrotate:
                rotateList = [0, 1, 2]
                degree = choice(rotateList)
                rawImg = cv2.rotate(rawImg, rotateCode=degree)  # 随机选择一个角度 （90,180,270）

            return rawImg  # 输入图像，标签图像（转换为tensor）

        elif self.method == "DeepSTORM":
            rawImgUp    = cv2.imread(rawImgUpFileName   , cv2.IMREAD_UNCHANGED)
            rawImgUp = rawImgUp.astype(np.float32)
            rawImgUp = normalize_maxmin(rawImgUp)
            rawImgUp = torch.tensor(rawImgUp).float()
            rawImgUp = rawImgUp.view(-1, rawImgUp.shape[0], rawImgUp.shape[1])

            if self.isrotate:
                rotateList = [0, 1, 2]
                degree = choice(rotateList)
                rawImgUp = cv2.rotate(rawImgUp, rotateCode=degree)

            return rawImgUp  # 输入图像，标签图像（转换为tensor）
        # heatMapImg  = cv2.imread(heatMapImgFileName , cv2.IMREAD_UNCHANGED)

        # heatMapImg  = heatMapImg.astype(np.float32)

        # 归一化
        # rawImg      = (rawImg   - rawImg.min())       / (rawImg.max()   - rawImg.min())
        # rawImgUp    = (rawImgUp - rawImgUp.min())     / (rawImgUp.max() - rawImgUp.min())
        # heatMapImg  = (heatMapImg - heatMapImg.min()) / (heatMapImg.max() - heatMapImg.min())

        # mean_val_test = rawImg.mean()
        # std_val_test  = rawImg.mean()
        # rawImg        = normalize_im(rawImg, mean_val_test, std_val_test)
        #
        # mean_val_test = rawImgUp.mean()
        # std_val_test  = rawImgUp.mean()
        # rawImgUp        = normalize_im(rawImgUp, mean_val_test, std_val_test)

        # 数据增强
        # plt.imshow(rawImg, cmap='gray')
        # plt.show()
        # plt.imshow(rawImgUp, cmap='gray')
        # plt.show()
        # plt.imshow(heatMapImg, cmap='gray')
        # plt.show()

            # heatMapImg  = cv2.rotate(heatMapImg,rotateCode=degree)

        # plt.imshow(rawImg, cmap='gray')
        # plt.show()
        # plt.imshow(rawImgUp, cmap='gray')
        # plt.show()
        # plt.imshow(heatMapImg, cmap='gray')
        # plt.show()

        # heatMapImg  = torch.tensor(heatMapImg).float()
        # heatMapImg  = heatMapImg.view(-1,heatMapImg.shape[0],heatMapImg.shape[1])


    def __len__(self):
        return len(self.fileNames)                       # 数据集的组数（1输入图像+1标签图像=1组）

if __name__ == "__main__":
    pathName = 'D:\project\Pro7-denseDL\data\simulation\data3\dataset_ROI19_F10000_density1.0'
    dataLoader= myDataLoader(pathName)

    next(iter(dataLoader))
    pass