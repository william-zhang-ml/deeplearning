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
                 padding: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1) -> None:
        """ Constructor.

        :param in_channels:  number of input channels
        :type  in_channels:  int
        :param out_channels: number of output channels
        :type  out_channels: int
        :param kernel_size:  kernel height and width, defaults to 3
        :type  kernel_size:  Union[int, Tuple[int, int]], optional
        :param stride:       stride height and width, defaults to 1
        :type  stride:       Union[int, Tuple[int, int]], optional
        :param padding:      row and col pixels to pad, defaults to 1
        :type  padding:      Union[int, Tuple[int, int]], optional
        :param groups:       channel groups in input, defaults to 1
        :type  groups:       int, optional
        """
        super(ConvBatchRelu, self).__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU())


class BatchReluConv(Sequential):
    """ Convnet building block: batchnorm, relu, conv (pre-activation).
        See https://arxiv.org/pdf/1603.05027.pdf.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False) -> None:
        """ Constructor.

        :param in_channels:  number of input channels
        :type  in_channels:  int
        :param out_channels: number of output channels
        :type  out_channels: int
        :param kernel_size:  kernel height and width, defaults to 3
        :type  kernel_size:  Union[int, Tuple[int, int]], optional
        :param stride:       stride height and width, defaults to 1
        :type  stride:       Union[int, Tuple[int, int]], optional
        :param padding:      row and col pixels to pad, defaults to 1
        :type  padding:      Union[int, Tuple[int, int]], optional
        :param groups:       channel groups in input, defaults to 1
        :type  groups:       int, optional
        :param bias:         whether conv layer uses bias, defaults to False
        :type  bias:         bool, optional
        """
        super(BatchReluConv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias))


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


class FeaturePyramidLinkR(Module):
    """ Feature pyramid network link (post-relu inputs).
        See: https://arxiv.org/pdf/1612.03144.pdf.
    """
    def __init__(self,
                 top_channels: int,
                 lat_channels: int,
                 mode: str = 'bilinear',
                 align_corners: bool = True) -> None:
        """ Constructor.

        :param top_channels:  num of channels in more abstract, more coarse map
        :type  top_channels:  int
        :param lat_channels:  num of channels in less abstract, more fine map
        :type  lat_channels:  int
        :param mode:          upsample interpolation approach,
                              defaults to 'bilinear'
        :type  mode:          str, optional
        :param align_corners: whether upsample output is aligned w/input,
                              defaults to True
        :type  align_corners: bool, optional
        """
        super(FeaturePyramidLinkR, self).__init__()
        self.top_channels = top_channels
        self.lat_channels = lat_channels
        self.mode = mode
        self.align_corners = align_corners
        self.nin = ConvBatchRelu(in_channels=lat_channels,
                                 out_channels=top_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.bottleneck = ConvBatchRelu(in_channels=top_channels,
                                        out_channels=lat_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, top: Tensor, lat: Tensor) -> Tensor:
        """ Compute multiscale feature map by upsampling and adding.

        :param top: more abstract, more coarse feature map
        :type  top: Tensor
        :param lat: less abstract, more fine feature map
        :type  lat: Tensor
        :return:    multiscale feature map
        :rtype:     Tensor
        """
        top_upsampled = F.interpolate(top,
                                      size=lat.shape[-2:],
                                      mode=self.mode,
                                      align_corners=self.align_corners)
        lat_features = self.nin(lat)
        return self.bottleneck(top_upsampled + lat_features)
