import torch
from myModel import myModel
import torch.optim as optim
import numpy as np
from dataset.dataLoad.SMLM_rawImgDataloader import myDataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Losses import MSEloss
import os,cv2

# train our model
def train_model(RawImgPath,savePath,saveTestPath):
    ## device initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## dataset preparation
    train_dataset       = myDataLoader(RawImgPath,isTrain=True,ratio=0.9,dataNums=2000)
    train_dataLoaders   = DataLoader(train_dataset,batch_size=16,shuffle=True)

    test_dataset        = myDataLoader(RawImgPath,isTrain=False,ratio=0.9,dataNums=2000)
    test_dataLoaders    = DataLoader(test_dataset, batch_size=4,shuffle=True)

    ## model
    # model = torch.load(os.path.join(savePath,'parameter_CNN_epoch%d.pkl'%200))
    model = myModel(num_channels=1)
    model.to(device)

    ## optimizer setting
    optimizer   = optim.Adam(model.parameters(), lr=0.0001) #0.001->0.0001

    ## start training
    train_loss_list  = []
    test_loss_list   = []

    bestTestLoss = 1e6
    EPOCH        = 30
    trainLoss    = 0
    testLoss     = 0
    for epoch in range(1, EPOCH + 1):
        for iteration,data in enumerate(train_dataLoaders,start=1):
            rawImgs,Heatmaps = data

            rawImgs     = rawImgs.cuda()
            Heatmaps    = Heatmaps.cuda()
            output      = model(rawImgs)
            trainLoss   = MSEloss(Heatmaps, output)
            optimizer.zero_grad()  # 梯度归零
            trainLoss.backward()
            optimizer.step()
            # print('%.6f'%trainLoss.cpu().detach().numpy())

        trainloss = trainLoss.cpu().detach().numpy()
        train_loss_list.append(trainloss)

        # validation
        for iteration,data in enumerate(test_dataLoaders,start=1):
            rawImgs,Heatmaps = data
            rawImgs         = rawImgs.cuda()
            Heatmaps        = Heatmaps.cuda()
            predict_y       = model(rawImgs)
            testLoss        = MSEloss(Heatmaps, predict_y)

        testloss = testLoss.cpu().detach().numpy()
        test_loss_list.append(testloss)

        if not os.path.exists(saveTestPath):
            os.mkdir(saveTestPath)
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        cv2.imwrite(os.path.join(saveTestPath, 'Heatmap_%d.tif'%epoch), Heatmaps[0, 0, :, :].cpu().detach().numpy().astype(np.float32))
        cv2.imwrite(os.path.join(saveTestPath, 'out_%d.tif'%epoch), predict_y[0, 0, :, :].cpu().detach().numpy().astype(np.float32))

        print('epoch:%d/%d,train_loss:%.6f,test_loss:%.10f' % (epoch, EPOCH, trainloss, testloss))

        torch.save(model, os.path.join(savePath,'parameter_CNN_epoch%d.pkl'%epoch))

        if testloss < bestTestLoss:
            bestTestLoss = testloss
            torch.save(model,os.path.join(savePath,'best.pkl'))

    plt.figure()
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['trainLoss', 'testLoss'])
    np.save(os.path.join(savePath,'trainLoss_CNN.npy') , np.array(train_loss_list))
    np.save(os.path.join(savePath,'testLoss_CNN.npy')  , np.array(test_loss_list))
    plt.show()

if __name__ == '__main__':
    rawImgPath      = r'D:\project\Pro7-mEDSR-STORM\code\python\demo\dataset\data\result'   # raw image pairs for training
    savePath        = r'D:\project\Pro7-mEDSR-STORM\code\python\demo\trainingResult'        # training result
    saveTestPath    = r'D:\project\Pro7-mEDSR-STORM\code\python\demo\trainingResult\test'   # test result when training

    train_model(rawImgPath,savePath,saveTestPath)


