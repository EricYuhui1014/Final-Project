"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from blurpool import *


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avg_pool = BlurPool(hidden_dim, filt_size=1, stride=stride), 
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )
    
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        if self.training:
            out_forward = torch.sign(x)
            mask1 = x < -1
            mask2 = x < 0
            mask3 = x < 1
            out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
            out = out_forward.detach() - out3.detach() + out3
            return out
        # print("in validation")
        return torch.sign(x)


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.out_chn = out_chn
        self.conv_inf = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        

    def forward(self, x):
    
        real_weights = self.weights.view(self.shape)
        # print("real")
        # print(real_weights[0,0,:,:])
        
        # scaling_factor = torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # print(scaling_factor.shape, flush=True)
        # print(scaling_factor[10:50,:,:,:])

        # scaling_factor = scaling_factor[0:1,:,:,:].item()
        scaling_factor = scaling_factor.view(-1,1,1)


        # scaling_factor = scaling_factor.detach()
        # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # print(binary_weights_no_grad)
        # print("binary")
        # print(binary_weights[0,0,:,:])
        # print("binary_d")
        # print(binary_weights_no_grad[0,0,:,:])
        if self.training:
            y = torch.mul(F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding), scaling_factor)
        else:
            self.conv_inf.weight = torch.nn.Parameter(binary_weights, requires_grad=False)
            
            y = torch.mul(self.conv_inf(x), scaling_factor)
            # print(self.conv_inf.weight)
            # print("using binarized weight for infrence")
        # print(x.shape)
        return y

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual_b(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual_b, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        layers = []
        if inp == hidden_dim:
            if stride==1:
                layers = [
                    # dw
                    # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2),
                    nn.BatchNorm2d(hidden_dim),

                    # h_swish() if use_hs else nn.ReLU(inplace=True),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # pw-linear
                    # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),

                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, oup, 1, 1, 0),
                    nn.BatchNorm2d(oup),
                ]
            else:
                layers = [
                    # dw
                    # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, hidden_dim, kernel_size, 1, (kernel_size - 1) // 2),
                    # BlurPool(hidden_dim, filt_size=kernel_size, stride=stride), 
                    nn.BatchNorm2d(hidden_dim),

                    # h_swish() if use_hs else nn.ReLU(inplace=True),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # pw-linear
                    # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    
                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, oup, 1, 1, 0),
                    nn.BatchNorm2d(oup),
                ]
        else:
            if stride==1:
                layers = [
                    # pw
                    # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    # nn.BatchNorm2d(hidden_dim),
                    # h_swish() if use_hs else nn.ReLU(inplace=True),
                    # dw
                    # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                    # nn.BatchNorm2d(hidden_dim),
                    # Squeeze-and-Excite
                    # SELayer(hidden_dim) if use_se else nn.Identity(),
                    # h_swish() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    # nn.BatchNorm2d(oup),

                                    # pw
                    # nn.BatchNorm3d(inp),
                    BinaryActivation(),
                    HardBinaryConv(inp, hidden_dim, 1, 1, 0),
                    nn.BatchNorm2d(hidden_dim),
                    # nn.PReLU(),
                    # dw
                    # nn.BatchNorm3d(hidden_dim),

                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2),
                    nn.BatchNorm2d(hidden_dim),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # nn.PReLU(),
                    # pw-linear
                    # nn.BatchNorm3d(hidden_dim),
                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, oup, 1, 1, 0),
                    nn.BatchNorm2d(oup),
                ]
            else:
                layers = [
                    # pw
                    # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    # nn.BatchNorm2d(hidden_dim),
                    # h_swish() if use_hs else nn.ReLU(inplace=True),
                    # dw
                    # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                    # nn.BatchNorm2d(hidden_dim),
                    # Squeeze-and-Excite
                    # SELayer(hidden_dim) if use_se else nn.Identity(),
                    # h_swish() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    # nn.BatchNorm2d(oup),

                                    # pw
                    # nn.BatchNorm3d(inp),
                    BinaryActivation(),
                    HardBinaryConv(inp, hidden_dim, 1, 1, 0),
                    nn.BatchNorm2d(hidden_dim),
                    # nn.PReLU(),
                    # dw
                    # nn.BatchNorm3d(hidden_dim),

                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, hidden_dim, kernel_size, 1, (kernel_size - 1) // 2),
                    # BlurPool(hidden_dim, filt_size=kernel_size, stride=stride), 
                    nn.BatchNorm2d(hidden_dim),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # nn.PReLU(),
                    # pw-linear
                    # nn.BatchNorm3d(hidden_dim),
                    BinaryActivation(),
                    HardBinaryConv(hidden_dim, oup, 1, 1, 0),
                    nn.BatchNorm2d(oup),
                ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode,if_binary=False, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1)] #<--for cifar-10/100 change s from 2 to 1

        # building inverted residual blocks
        block = InvertedResidual
        if if_binary:
            block = InvertedResidual_b
        else:
            block = InvertedResidual


        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        
        # self.avgpool = BlurPool(576, filt_size=4, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 1], #<--for cifar-10/100 change s from 2 to 1
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)