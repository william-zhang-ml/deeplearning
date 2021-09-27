from typing import Union, Tuple
import torch.nn as nn
from torch.nn import Sequential


class ConvBatchRelu(Sequential):
    """ Fundamental convnet building block: conv-batchnorm-relu. """
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
