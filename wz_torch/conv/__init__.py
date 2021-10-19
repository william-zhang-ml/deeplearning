""" Unsorted deep learning utils for computer vision. """
from typing import Union, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Sequential, Module
import torch.nn.functional as F


class ConvBatchRelu(Sequential):
    """ Fundamental convnet building block: conv, batchnorm, relu. """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1) -> None:
        """ Constructor.

        :param in_channels:  number of input channels
        :type  in_channels:  int
        :param out_channels: number of output channels
        :type  out_channels: int
        :param kernel_size:  kernel height and width, defaults to 3
        :type  kernel_size:  Union[int, Tuple[int, int]], optional
        :param stride:       stride height and width, defaults to 1
        :type  stride:       Union[int, Tuple[int, int]], optional
        :param padding:      number of row and col pixels to pad, defaults to 1
        :type  padding:      Union[int, Tuple[int, int]], optional
        """
        super(ConvBatchRelu, self).__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU())


class ResidualBottleneck(Module):
    """ Residual block: 1 x 1, 3 x 3, 1 x 1, full pre-activation.
        See https://arxiv.org/pdf/1603.05027.pdf Fig 4e.
    """
    def __init__(self,
                 in_channels: int,
                 bottle_channels: int) -> None:
        """ Constructor.

        :param in_channels:     number of input channels
        :type  in_channels:     int
        :param bottle_channels: number of bottleneck channels (< in_channels)
        :type  bottle_channels: int
        """
        super(ResidualBottleneck, self).__init__()
        self.residual = Sequential(
            nn.BatchNorm2d(in_channels),       # preactivation
            nn.ReLU(),                         # preactivation
            ConvBatchRelu(
                in_channels=in_channels,
                out_channels=bottle_channels,
                kernel_size=1,
                stride=1,
                padding=0),                    # 1 x 1
            ConvBatchRelu(
                in_channels=bottle_channels,
                out_channels=bottle_channels,
                kernel_size=3,
                stride=1,
                padding=1),                    # 3 x 3
            nn.Conv2d(
                in_channels=bottle_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0))                    # 1 x 1

    def forward(self, inp: Tensor) -> Tensor:
        """ Compute residual and add to main path.

        :param inp: feature map (N, D, H, W)
        :type  inp: Tensor
        :return:    residual-adjusted feature map
        :rtype:     Tensor
        """
        return inp + self.residual(inp)


class GapSoftmax(Module):
    """ Classification head: global average pooling, softmax.
        See: https://arxiv.org/pdf/1312.4400.pdf Section 3.2.
    """
    def __init__(self) -> None:
        """ Constructor. """
        super(GapSoftmax, self).__init__()

    def forward(self, inp: Tensor) -> Tensor:
        """ Compute class confidence scores from average feature map.

        :param inp: feature map (N, D, H, W)
        :type  inp: Tensor
        :return:    class confidence scores (N, D)
        :rtype:     Tensor
        """
        ave = inp.mean(dim=-1).mean(dim=-1)
        return F.softmax(ave, dim=-1)


class GapLinearSoftmax(Module):
    """ Classification head: global average pooling, softmax.
        See: https://arxiv.org/pdf/1312.4400.pdf Section 3.2.
    """
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 make_cam: bool = False) -> None:
        """ Constructor.

        :param in_features: number of inputs
        :type  in_features: int
        :param num_classes: number of target classes
        :type  num_classes: int
        :param make_cam:    whether to make class activation map for outputs,
                            defaults to False
        :type  make_cam:    bool, optional
        """
        super(GapLinearSoftmax, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.make_cam = make_cam

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        """ Compute class confidence scores from average feature map.

        :param inp: feature map (N, in_features, H, W)
        :type  inp: Tensor
        :return:    class confidence scores (N, num_classes),
                    class activation map (N, num_classes, H, W)
        :rtype:     Tuple[Tensor, Tensor]
        """
        num_samp, num_maps, height, width = inp.shape
        ave = inp.mean(dim=-1).mean(dim=-1)
        preact = self.linear(ave)

        if self.make_cam:
            featmap_vecs = inp.view(num_samp, num_maps, height * width)
            weight = self.linear.weight.detach().unsqueeze(0)

            # (1, num_classes, num_maps) @ (num_samp, num_maps, height * width)
            # -> (num_samp, num_classes, height * width)
            cam = (weight @ featmap_vecs).view(num_samp, -1, height, width)
        else:
            cam = None
        return F.softmax(preact, dim=-1), cam


class GapLinearSigmoid(Module):
    """ Classification head: global average pooling, sigmoid.
        See: https://arxiv.org/pdf/1312.4400.pdf Section 3.2.
    """
    def __init__(self,
                 in_features: int,
                 make_cam: bool = False) -> None:
        """ Constructor.

        :param in_features: number of inputs
        :type  in_features: int
        :param make_cam:    whether to make class activation map for outputs,
                            defaults to False
        :type  make_cam:    bool, optional
        """
        super(GapLinearSigmoid, self).__init__()
        self.linear = nn.Linear(in_features, 1)
        self.make_cam = make_cam

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        """ Compute confidence score from average feature map.

        :param inp: feature map (N, in_features, H, W)
        :type  inp: Tensor
        :return:    class confidence scores (N, 1),
                    class activation map (N, 1, H, W)
        :rtype:     Tuple[Tensor, Tensor]
        """
        num_samp, num_maps, height, width = inp.shape
        ave = inp.mean(dim=-1).mean(dim=-1)
        preact = self.linear(ave)

        if self.make_cam:
            featmap_vecs = inp.view(num_samp, num_maps, height * width)
            weight = self.linear.weight.detach().unsqueeze(0)

            # (1, num_classes, num_maps) @ (num_samp, num_maps, height * width)
            # -> (num_samp, num_classes, height * width)
            cam = (weight @ featmap_vecs).view(num_samp, -1, height, width)
        else:
            cam = None
        return torch.sigmoid(preact), cam
