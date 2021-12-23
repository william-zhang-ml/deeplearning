""" Residual network implementations.

ResNeXt https://arxiv.org/pdf/1611.05431.pdf (Apr 2017).
"""
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from . import ConvBatchRelu


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
