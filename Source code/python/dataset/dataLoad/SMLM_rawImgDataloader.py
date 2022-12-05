from torch.utils.data import Dataset
from imageUtils import normalize_maxmin
from random import choice
import os
import cv2
import numpy as np
import torch
class myDataLoader(Dataset):
    def __init__(self,path,isTrain=True,ratio=0.95,dataNums = 1000,IsallImgUsed = False,isrotate = True):
        self.path = path
        self.RawImgPath     = os.path.join(self.path,'rawImg')
        self.RawImgUpPath   = os.path.join(self.path,'rawImgUp')
        self.HeatmapImgPath = os.path.join(self.path,'HeatmapImg')

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
        heatMapImgFileName  = os.path.join(self.HeatmapImgPath  ,self.fileNames[index])  # heatmap路径

        # 读取图像
        rawImg      = cv2.imread(rawImgFileName     , cv2.IMREAD_UNCHANGED)
        heatMapImg  = cv2.imread(heatMapImgFileName , cv2.IMREAD_UNCHANGED)

        rawImg      = rawImg.astype(np.float32)
        heatMapImg  = heatMapImg.astype(np.float32)

        if self.isrotate:
            rotateList = [0,1,2]
            degree = choice(rotateList)
            rawImg      = cv2.rotate(rawImg,rotateCode=degree)              # 随机选择一个旋转角度 （90,180,270）
            heatMapImg  = cv2.rotate(heatMapImg,rotateCode=degree)

        # plt.imshow(rawImg, cmap='gray')
        # plt.show()
        # plt.imshow(rawImgUp, cmap='gray')
        # plt.show()
        # plt.imshow(heatMapImg, cmap='gray')
        # plt.show()

        rawImg      = normalize_maxmin(rawImg)

        # 转换为pytorch dataloader使用的格式
        rawImg      = torch.tensor(rawImg).float()
        rawImg      = rawImg.view(-1,rawImg.shape[0],rawImg.shape[1])

        heatMapImg  = torch.tensor(heatMapImg).float()
        heatMapImg  = heatMapImg.view(-1,heatMapImg.shape[0],heatMapImg.shape[1])

        return rawImg, heatMapImg              # 输入图像，标签图像（转换为tensor）

    def __len__(self):
        return len(self.fileNames)                       # 数据集的组数（1输入图像+1标签图像=1组）

if __name__ == "__main__":
    pathName = 'D:\project\Pro7-denseDL\data\simulation\data3\dataset_ROI19_F10000_density1.0'
    dataLoader= myDataLoader(pathName)

    next(iter(dataLoader))
    pass