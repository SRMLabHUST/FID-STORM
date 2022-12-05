import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import os
from dataset.dataLoad.SMLM_rawImgDataloader_test import myDataLoaderTest
from datetime import datetime

def test():
    ## Input parameters
    rawImgPath  = r'D:\project\Pro7-mEDSR-STORM\code\python\demo\dataset\data\result'     # raw image directory
    modelPath   = r'D:\project\Pro7-mEDSR-STORM\code\python\demo\trainingResult'          # model path, best.pkl will be loaded
    savePath    = r'D:\project\Pro7-mEDSR-STORM\code\python\demo\trainingResult'          # save path, output images and timeList will be saved
    subDir      = r'output'     # sub directory, using for saving output images

    test_dataset        = myDataLoaderTest(rawImgPath,isTrain=False,ratio=0.9,IsallImgUsed=True,isrotate=False,dataNums=4000,method="ours")
    test_dataLoaders    = DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=0)

    ## model
    model   = torch.load(os.path.join(modelPath, 'best.pkl'))  # load best.pkl model
    model.eval()

    for iteration,data in enumerate(test_dataLoaders,start=0):
        rawImgs = data

        rawImgs = rawImgs.cuda()

        with torch.no_grad():
            predict_y = model(rawImgs)

        fileIndex = test_dataset.fileNames[iteration]
        fileIndex = fileIndex[:-4]

        # cv2.imwrite(os.path.join(savePath,'rawImg.tif'),rawImgUps[0,0,:,:].cpu().detach().numpy())
        # cv2.imwrite(os.path.join(savePath,'Heatmap.tif'),Heatmaps[0,0,:,:].cpu().detach().numpy())
        if not os.path.exists(os.path.join(savePath,subDir)):
            os.mkdir(os.path.join(savePath,subDir))
        # cv2.imwrite(os.path.join(savePath,'out_%d.tif'%int(fileIndex)),predict_y[0,0,:,:].cpu().detach().numpy())
        cv2.imwrite(os.path.join(os.path.join(savePath,subDir), 'out_%d.tif' % int(fileIndex)),
                    predict_y[0, 0, :, :].cpu().detach().numpy())
        # print(100,'/',iteration)

if __name__ == '__main__':
    test()               # inference

