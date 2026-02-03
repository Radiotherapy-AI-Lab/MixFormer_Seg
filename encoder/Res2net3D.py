import math

import torch
from torch import nn


class Bottle2neck3D(nn.Module):
    # 移除类级别的expansion，改为实例属性
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal',
                 expansion=4):  # 添加expansion参数
        super(Bottle2neck3D, self).__init__()
        self.expansion = expansion  # 实例属性
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv3d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=3, stride=stride, padding=1)

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm3d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width * scale, planes * self.expansion, kernel_size=1, bias=False)  # 使用实例属性
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    # forward方法保持不变
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net3D(nn.Module):
    def __init__(self,  block, layers, ch_in=1, dim=64, baseWidth=26, scale=4, num_classes=1000,
                 expansion=4):  # 添加expansion参数
        # self.cfg = cfg
        self.inplanes = dim
        super(Res2Net3D, self).__init__()
        self.baseWidth = baseWidth
        self.in_channel = ch_in
        self.dim = dim
        self.scale = scale
        self.conv1 = nn.Conv3d(self.in_channel, self.dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.dim, layers[0], expansion=expansion)  # 传递expansion
        self.layer2 = self._make_layer(block, self.dim * 2, layers[1], stride=2, expansion=expansion)
        self.layer3 = self._make_layer(block, self.dim * 4, layers[2], stride=2, expansion=expansion)
        self.layer4 = self._make_layer(block, self.dim * 8, layers[3], stride=2, expansion=expansion)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * expansion, num_classes)  # 使用传入的expansion
        self.dropout = nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, expansion=4):  # 添加expansion参数
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:  # 使用传入的expansion
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * expansion,  # 使用expansion
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * expansion),  # 使用expansion
                nn.Dropout3d(0.2)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale,
                            expansion=expansion))  # 传递expansion
        self.inplanes = planes * expansion  # 使用expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale,
                                expansion=expansion))  # 传递expansion

        return nn.Sequential(*layers)



    # forward方法保持不变
    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return features


def res2net50_3d(**kwargs):
    # 允许通过kwargs传递expansion参数（例如expansion=8）
    model = Res2Net3D(Bottle2neck3D, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model


def test_res2net3d():
    # 测试不同expansion值
    for expansion in [1, 8]:  # 测试默认值和自定义值
        # 初始化模型（指定expansion）
        model = res2net50_3d(num_classes=10, dim=96, expansion=expansion)

        # 生成随机3D输入数据 [batch, channels, depth, height, width]
        input_data = torch.randn(1, 1, 64, 224, 224)  # 2样本，3通道，32x32x32体积

        # 前向传播
        output = model(input_data)

        for i in output:
            print(i.size())

# if __name__ == "__main__":
#
#     test_res2net3d()  # 新增测试
