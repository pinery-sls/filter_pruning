from __future__ import absolute_import
import math
import torch.nn as nn
import numpy as np
import torch

__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""




class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # print('cfg[0] = ', cfg[0])
        # print('cfg[1] = ', cfg[1])
        # print('cfg[2] = ', cfg[2])
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)

        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, layers, number_class=1000, cfg=None):
        super(resnet, self).__init__()
        # assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
        #
        # n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [[64, 64, 64], [256, 64, 64] * (layers[0] - 1), [256, 128, 128], [512, 128, 128] * (layers[1] - 1),
                   [512, 256, 256], [1024, 256, 256] * (layers[2] - 1), [1024, 512, 512],
                   [2048, 512, 512] * (layers[3] - 1), [2048]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 64

        cfg_start = 0
        cfg_end = 3*layers[0]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # print('33333333', cfg[0:3*n])
        self.layer1 = self._make_layer(block, 64, layers[0], cfg = cfg[cfg_start:cfg_end])

        cfg_start += 3 * layers[0]
        cfg_end += 3 * layers[1]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg = cfg[cfg_start:cfg_end], stride=2)

        cfg_start += 3 * layers[1]
        cfg_end += 3 * layers[2]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg = cfg[cfg_start:cfg_end], stride=2)

        cfg_start += 3 * layers[2]
        cfg_end += 3 * layers[3]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[cfg_start:cfg_end], stride=2)

        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.select = channel_selection(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7,stride=1)

        self.fc = nn.Linear(cfg[-1], number_class)

        # if dataset == 'cifar10':
        #     self.fc = nn.Linear(cfg[-1], 10)
        # elif dataset == 'cifar100':
        #     self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        # print('1111111111', cfg[0:3])
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # print('222222222',cfg[3*i: 3*(i+1)])
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__=="__main__":
    model = resnet([3, 4, 6, 3])
    print(model)
    print("resnet50 have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
