""" Residual network implementations.

Deep Residual Learning https://arxiv.org/pdf/1512.03385.pdf (Dec 2015).
Identiy Mappings in ResNets https://arxiv.org/pdf/1603.05027.pdf (Jul 2016)
ResNeXt https://arxiv.org/pdf/1611.05431.pdf (Apr 2017).
"""
from typing import Tuple, Union
import torch
import torch.nn as nn
from . import BatchReluConv


class Residual(nn.Module):
    """ Residual block: 2 idential blocks (pre-activation).
        See https://arxiv.org/pdf/1512.03385.pdf
        and https://arxiv.org/pdf/1603.05027.pdf.
    """
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1) -> None:
        """ Constructor.

        :param channels:     number of input channels
        :type  channels:     int
        :param kernel_size:  kernel height and width, defaults to 3
        :type  kernel_size:  Union[int, Tuple[int, int]], optional
        :param stride:       stride height and width, defaults to 1
        :type  stride:       Union[int, Tuple[int, int]], optional
        :param padding:      row and col pixels to pad, defaults to 1
        :type  padding:      Union[int, Tuple[int, int]], optional
        """
        super(Residual, self).__init__()
        self.residual = nn.Sequential(
            BatchReluConv(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            BatchReluConv(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ Compute residual and add to main path.

        :param inp: feature map (N, C, H, W)
        :type  inp: torch.Tensor
        :return:    residual-adjusted feature map (N, C, H, W)
        :rtype:     torch.Tensor
        """
        return inp + self.residual(inp)


class ResidualDownsample(nn.Module):
    """ Residual block: 3 stride 2 + user block (pre-activation).
        See https://arxiv.org/pdf/1512.03385.pdf
        and https://arxiv.org/pdf/1603.05027.pdf.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1) -> None:
        """ Constructor.

        :param in_channels:  number of input channels
        :type  in_channels:  int
        :param out_channels: number of output channels, defaults to None (x2)
        :type  out_channels: int, optional
        :param kernel_size:  kernel height and width, defaults to 3
        :type  kernel_size:  Union[int, Tuple[int, int]], optional
        :param stride:       stride height and width, defaults to 1
        :type  stride:       Union[int, Tuple[int, int]], optional
        :param padding:      row and col pixels to pad, defaults to 1
        :type  padding:      Union[int, Tuple[int, int]], optional
        """
        super(ResidualDownsample, self).__init__()
        out_channels = out_channels or 2 * in_channels
        self.residual = nn.Sequential(
            BatchReluConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            BatchReluConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))
        self.projection = BatchReluConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ Compute residual and add to main path.

        :param inp: feature map (N, Cin, H, W)
        :type  inp: torch.Tensor
        :return:    residual-adjusted, downsampled feature map (N, Cout, H, W)
        :rtype:     torch.Tensor
        """
        return self.projection(inp) + self.residual(inp)


class Residual131(nn.Module):
    """ Residual block: 1 x 1, user block, 1 x 1 (pre-activation).
        See https://arxiv.org/pdf/1603.05027.pdf Fig 4e.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1) -> None:
        """ Constructor.

        :param in_channels:     number of input channels
        :type  in_channels:     int
        :param hidden_channels: number of hidden residual channels
        :type  hidden_channels: int
        :param kernel_size:     kernel height and width, defaults to 3
        :type  kernel_size:     Union[int, Tuple[int, int]], optional
        :param stride:          stride height and width, defaults to 1
        :type  stride:          Union[int, Tuple[int, int]], optional
        :param padding:         row and col pixels to pad, defaults to 1
        :type  padding:         Union[int, Tuple[int, int]], optional
        :param groups:          number of input channel groups, defaults to 1
        :type  groups:          int, optional
        """
        super(Residual131, self).__init__()
        self.residual = nn.Sequential(
            BatchReluConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchReluConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups),
            BatchReluConv(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ Compute residual and add to main path.

        :param inp: feature map (N, C, H, W)
        :type  inp: torch.Tensor
        :return:    residual-adjusted feature map (N, C, H, W)
        :rtype:     torch.Tensor
        """
        return inp + self.residual(inp)


class ResNeXt(Residual131):
    """ ResNeXt block: 1 x 1, user block, 1 x 1 (pre-activation).
        See https://arxiv.org/pdf/1611.05431.pdf Fig 3c.
    """
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
        super(ResNeXt, self).__init__(
            in_channels=in_channels,
            hidden_channels=embed_channels * cardinality,
            groups=cardinality
        )
