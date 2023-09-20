import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        # self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DilatedSpatialPath(nn.Module):
    def __init__(self, inplanes, outplanes, D_list):
        super(DilatedSpatialPath, self).__init__()
        d_list = []
        self.hoist = False
        if inplanes != outplanes:
            self.hoist = True
            self.hconv = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0)
        for i in D_list:
            layer = nn.Sequential(
                nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=i, dilation=i, bias=False),
                # BatchNorm2d(outplanes, momentum=bn_mom),
                nn.ReLU(inplace=True)
            )
            d_list.append(layer)
        self.conv = nn.Sequential(*d_list)

    def forward(self, x):
        if self.hoist:
            x = self.hconv(x)
        out = self.conv(x)
        return out



class TAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(TAPPM, self).__init__()

        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                    )
        self.scale11 = nn.Sequential(nn.AvgPool2d(kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
                                     BatchNorm2d(branch_planes, momentum=bn_mom),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 3), padding=(1, 1),
                                               bias=False),
                                     )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                    )
        self.scale22 = nn.Sequential(nn.AvgPool2d(kernel_size=(9, 1), stride=(4, 1), padding=(4, 0)),
                                     BatchNorm2d(branch_planes, momentum=bn_mom),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 3), padding=(1, 1),
                                               bias=False),
                                     )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 8), padding=(0, 8)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                    )
        self.scale33 = nn.Sequential(nn.AvgPool2d(kernel_size=(17, 1), stride=(8, 1), padding=(8, 0)),
                                     BatchNorm2d(branch_planes, momentum=bn_mom),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 3), padding=(1, 1),
                                               bias=False)
                                     )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, padding=0, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, padding=0, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, padding=0, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )
        self.atten = BiAttention(outplanes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        s1 = self.scale1(x)
        s11 = self.scale11(s1)
        x_list.append((F.interpolate(s11, size=[height, width], mode='bilinear', align_corners=False)))
        s2 = self.scale2(x)
        s22 = self.scale22(s2)
        x_list.append(((F.interpolate(s22, size=[height, width], mode='bilinear', align_corners=False))))
        s3 = self.scale3(x)
        s33 = self.scale33(s3)
        x_list.append((F.interpolate(s33, size=[height, width], mode='bilinear', align_corners=False)))
        x_list.append(F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=False))

        # out = self.atten(self.compression(torch.cat(x_list, 1)), self.shortcut(x))
        out = self.atten(self.shortcut(x), self.compression(torch.cat(x_list, 1)))
        return out


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None, drop_rate=0):
        super(segmenthead, self).__init__()
        # self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        # self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv1(self.relu(x))
        # x = self.dropout(x)
        # out = self.conv2(self.relu(self.bn2(x)))
        out = self.conv2(self.relu(x))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear', align_corners=False)

        return out


class BiAttention(nn.Module):
    def __init__(self, in_channels):
        super(BiAttention, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        # x1: encoder feature map, x2: decoder feature map
        x3 = self.relu(x1 + x2)
        x3 = self.conv3(x3)
        alpha = torch.sigmoid(x3)
        x4 = x1 * alpha
        x5 = x2 * (1 - alpha)
        out = x4 + x5
        return out


class Decoder(nn.Module):
    def __init__(self, inplanes, outplanes, mode='add'):
        super(Decoder, self).__init__()
        self.mode = mode
        inp = inplanes
        if mode == 'cat':
            inp = inplanes * 2
        self.conv_H = nn.Conv2d(inp, inplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_W = nn.Conv2d(inp, inplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_out = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, d1, d2, x, y):
        y = F.interpolate(
            self.relu(y),
            size=[x.size(2), x.size(3)],
            mode='bilinear', align_corners=False)
        if self.mode == 'cat':
            d = torch.cat((d1, d2), 1)
        else:
            d = d1 + d2
        atten_h = torch.sigmoid(self.conv_H(d))
        atten_w = torch.sigmoid(self.conv_W(d))
        atten = atten_h + atten_w
        y_s = y * (1 - atten)
        x_s = x * atten
        out = x_s + x + y_s + y
        out = self.conv_out(out)
        return out


class DualResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=True,
                 deploy=False):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment
        self.relu = nn.ReLU(inplace=False)
        # Encoder
        self.stom = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            # BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            # BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[1], stride=1)
        # Segmatic Path(Encoder)
        self.layer4 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer5 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer6 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = TAPPM(planes * 16, spp_planes, planes * 4)
        # Spatial Path
        self.dp1 = DilatedSpatialPath(planes * 2, planes * 4, D_list=[1, 2, 5])
        self.dp2 = DilatedSpatialPath(planes * 4, planes * 4, D_list=[7, 13])
        # Decoder
        self.decoder = Decoder(planes * 4, planes * 2, mode='add')
        self.final_layer = segmenthead(planes * 2, head_planes, num_classes)

        self.compression = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            # BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            # BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.down5 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            # BatchNorm2d(planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            # BatchNorm2d(planes * 8, momentum=bn_mom),
        )
        if self.augment:
            self.seghead_extra = segmenthead(planes * 4, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 4, planes, 1)
            self.seghead_d2 = segmenthead(planes * 4, planes, 1)

        # ImageNet预训练
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1000),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        x = self.stom(x)
        x = self.layer1(x)
        x = self.layer2(self.relu(x))
        x3 = self.layer3(self.relu(x))
        dp1 = self.dp1(self.relu(x3[:, 0:x3.shape[1] // 2, :, :]))
        x4 = self.layer4(self.relu(x3[:, x3.shape[1] // 2:x3.shape[1], :, :]))
        x = x4 + self.down4(dp1)
        dp1 = dp1 + F.interpolate(
            self.relu(x4),
            size=[height_output, width_output],
            mode='bilinear', align_corners=False)

        x = self.layer5(self.relu(x))
        dp1 = x3 + dp1
        dp2 = self.dp2(dp1)
        x = x + self.down5(self.relu(dp2))
        dp = dp2 + F.interpolate(
            self.compression(self.relu(x)),
            size=[height_output, width_output],
            mode='bilinear', align_corners=False)
        x = self.layer6(self.relu(x))
        x = self.spp(self.relu(x))
        x = self.decoder(dp1, dp2, dp, x)

        x = self.final_layer(x)
        if self.augment:
            x_s1 = self.seghead_extra(x4)
            x_d1 = self.seghead_d(dp)
            return [x_s1, x, x_d1]
        else:
            return x


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


def DualResNet_imagenet(cfg, pretrained=False):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64,
                       augment=True, deploy=False)

    if pretrained:
        print("Using pretrained")
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        model_dict = model.state_dict()
        pretrained_state = {k[6:]: v for k, v in pretrained_state.items() if
                            (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_state)
        model.load_state_dict(model_dict, strict=False)
    return model


def get_imagenet_model():
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, planes=32, spp_planes=128, head_planes=64,
                       augment=False)
    return model


def get_seg_model(cfg, imgnet_pretrained, **kwargs):
    model = DualResNet_imagenet(cfg, pretrained=imgnet_pretrained)
    return model


def get_pred_model():
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64,
                       augment=False, deploy=True)

    return model


import time

if __name__ == '__main__':
    device = torch.device('cuda')
    model = get_pred_model()
    import time
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
