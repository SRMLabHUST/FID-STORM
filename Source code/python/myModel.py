# %%
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary
import math

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2, stride=1,bias=False)   # 卷积 layer1 , 用于提取浅层特征
        self.norm_layer = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.lrelu(self.norm_layer(self.conv(x)))

class myModel(nn.Module):
    def __init__(self, num_channels=1):
        super(myModel, self).__init__()

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=3 // 2, bias=False),     # 卷积 layer1 , 用于提取浅层特征
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.resBlocks = self.residualBlocks(Conv_ReLU_Block,num_of_layer=10)

        self.last_part = nn.Sequential(                                                 # 卷积
            ########################shuffle 卷积########################
            nn.Conv2d(64, num_channels * (8 ** 2), kernel_size=3, padding=3 // 2),
            # nn.Conv2d(64, 16 * 4**2, kernel_size=3, padding=3 // 2, bias=False),      # 放大8倍
            nn.PixelShuffle(4),
            # nn.Conv2d(64, 64 * 4, kernel_size=3, padding=3 // 2),
            # nn.PixelShuffle(2),
            nn.Conv2d(4, 1 * 2**2, kernel_size=3,  padding=3 // 2, bias=False),
            nn.PixelShuffle(2)
        )

        # self.conv_out = nn.Conv2d(16, 1, kernel_size=3,  padding=3 // 2)

        self.convup1 = nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2, stride=1,bias=False)
        self.convup2 = nn.Conv2d(32, 16, kernel_size=3, padding=3 // 2, stride=1,bias=False)
        self.convup3 = nn.Conv2d(16, 1,  kernel_size=3, padding=3 // 2, stride=1,bias=False)

        self.normlayerUp1 = nn.BatchNorm2d(32)
        self.normlayerUp2 = nn.BatchNorm2d(16)

        self._initialize_weights()      # 模型参数初始化


    def residualBlocks(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    # nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    # nn.init.zeros_(m.bias.data)

    def forward(self, x):
        out = self.first_part(x)
        resblock = self.resBlocks(out)
        out = torch.add(out, resblock)

        up1      = nn.Upsample(scale_factor=2)(out)     # 使用scale_factor指定倍率
        convup1  = self.convup1(up1)  # 卷积 layer1 , 用于提取浅层特征
        # convup1 = self.normlayerUp1(convup1)
        out = nn.LeakyReLU(inplace=True)(convup1)

        up2      = nn.Upsample(scale_factor=2)(out)
        convup2  = self.convup2(up2)  # 卷积 layer1 , 用于提取浅层特征
        # convup2 = self.normlayerUp2(convup2)
        out = nn.LeakyReLU(inplace=True)(convup2)

        up3      = nn.Upsample(scale_factor=2)(out)
        convup3  = self.convup3(up3)  # 卷积 layer1 , 用于提取浅层特征
        out = nn.LeakyReLU(inplace=True)(convup3)

        # out = self.conv_out(out)
        return out

from datetime import datetime
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = myModel(num_channels=1)
    model.to(device)
    input = torch.zeros((20,1,200,200))
    input = input.cuda()

    # model.eval()
    # t0 = datetime.now()
    # with torch.no_grad():
    #     out = model(input)
    # t1 = datetime.now()
    #
    # print(t1-t0,'s')
    # print(model(img_tensor).shape)
    summary(model,input_size=(1,256,256),device = 'cuda')