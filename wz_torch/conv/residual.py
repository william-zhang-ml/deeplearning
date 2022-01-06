""" Residual network implementations.

ResNeXt https://arxiv.org/pdf/1611.05431.pdf (Apr 2017).
"""
from typing import Tuple, Union
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from . import ConvBatchRelu, BatchReluConv


class ResidualBottleneck(Module):
    """ Residual block: 1 x 1, 3 x 3, 1 x 1, full pre-activation.
        See https://arxiv.org/pdf/1603.05027.pdf Fig 4e.
    """
    def __init__(self,
                 in_channels: int,
                 bottle_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1) -> None:
        """ Constructor.

        :param in_channels:     number of input channels
        :type  in_channels:     int
        :param bottle_channels: number of bottleneck channels (< in_channels)
        :type  bottle_channels: int
        :param kernel_size:     kernel height and width, defaults to 3
        :type  kernel_size:     Union[int, Tuple[int, int]], optional
        :param stride:          stride height and width, defaults to 1
        :type  stride:          Union[int, Tuple[int, int]], optional
        :param padding:         row and col pixels to pad, defaults to 1
        :type  padding:         Union[int, Tuple[int, int]], optional
        """
        super(ResidualBottleneck, self).__init__()
        self.residual = nn.Sequential(
            BatchReluConv(
                in_channels=in_channels,
                out_channels=bottle_channels,
                kernel_size=1,
                stride=1,
                padding=0),                    # 1 x 1
            BatchReluConv(
                in_channels=bottle_channels,
                out_channels=bottle_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),              # 3 x 3
            BatchReluConv(
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


class ResNeXt(Module):
    """ ResNeXt residual layer w/pre-activation.

        Aggregated Residual Transformations for Deep Neural Network - Fig 3c
        https://arxiv.org/pdf/1611.05431.pdf """
    def __init__(self,
                 in_channels: int,
                 embed_channels: int,
                 cardinality: int) -> None:
        """ Constructor.

        :param in_channels:    number of input channels
        :type  in_channels:    int
        :param embed_channels: embedding dimension of each residual branch
        :type  embed_channels: int
        :param cardinality:    number of residual branches
        :type  cardinality:    int
        """
        super(ResNeXt, self).__init__()
        width = embed_channels * cardinality
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            ConvBatchRelu(in_channels=in_channels,
                          out_channels=width,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            ConvBatchRelu(in_channels=width,
                          out_channels=width,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=cardinality),
            nn.Conv2d(in_channels=width,
                      out_channels=in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )

    def forward(self, inp: Tensor) -> Tensor:
        """ Compute residual and add to main path.

        :param inp: feature map (N, D, H, W)
        :type  inp: Tensor
        :return:    residual-adjusted feature map (N, D, H, W)
        :rtype:     Tensor
        """
        return inp + self.residual(inp)
