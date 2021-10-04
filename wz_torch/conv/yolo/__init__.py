from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module


class YoloHead(Module):
    """ Yolo version 1 - 3 prediction head. """
    def __init__(self,
                 in_channels: int,
                 num_class: int,
                 prior: Tuple[float, float],
                 use_softmax: bool = False) -> None:
        """ Constructor.

        :param in_channels: expected input feature map depth (D)
        :type  in_channels: int
        :param num_class:   number of target classes (K)
        :type  num_class:   int
        :param prior:       prior box associated with this head
        :type  prior:       Tuple[float, float]
        :param use_softmax: whether to classify obj w/softmax or sigmoid,
                            defaults to False
        :type  use_softmax: bool, optional
        """
        super(YoloHead, self).__init__()
        self.register_buffer('num_class', torch.tensor(num_class))
        self.register_buffer('prior', torch.tensor(prior))
        self.detect = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=1,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())            # confidence that pred cell contains obj
        self.center_offset = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=2,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid())            # obj center location within pred cell
        self.regression = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=2,
                      kernel_size=1,
                      padding=0))    # obj size relative to prior box
        if use_softmax:
            self.classify = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=num_class,
                          kernel_size=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_class),
                nn.Softmax(dim=1))   # obj class
        else:
            self.classify = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=num_class,
                          kernel_size=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_class),
                nn.Sigmoid())        # obj class

    def forward(self, inp: Tensor) -> Tensor:
        """ Detect, localize, and classify objects.

        :param inp: input feature map (B, D, H, W)
        :type  inp: Tensor
        :return:    Yolo prediction grid (B, 5 + K, H, W)
        :rtype:     Tensor
        """
        detect = self.detect(inp)         # B, 1, H, W
        offset = self.center_offset(inp)  # B, 2, H, W
        regr = self.regression(inp)       # B, 2, H, W
        class_conf = self.classify(inp)   # B, K, H, W
        return torch.cat([detect, offset, regr, class_conf], dim=1)


class YoloLayer(Module):
    """ Parallel Yolo version 1 - 3 prediction heads. """
    def __init__(self,
                 in_channels: int,
                 num_class: int,
                 priors: Tuple[float, float],
                 use_softmax: bool = False) -> None:
        """ Constructor.

        :param in_channels: expected input feature map depth (D)
        :type  in_channels: int
        :param num_class:   number of target classes (K)
        :type  num_class:   int
        :param priors:      prior box associated with this head
        :type  priors:      Tuple[Tuple[float, float]]
        :param use_softmax: whether to classify obj w/softmax or sigmoid,
                            defaults to False
        :type  use_softmax: bool, optional
        """
        super(YoloLayer, self).__init__()
        self.heads = nn.ModuleList([
            YoloHead(in_channels=in_channels,
                     num_class=num_class,
                     prior=p,
                     use_softmax=use_softmax)
            for p in priors
        ])

    def forward(self, inp: Tensor) -> Tensor:
        """ Detect, localize, and classify objects.

        :param inp: input feature map (B, D, H, W)
        :type  inp: Tensor
        :return:    Yolo prediction grid (P, B, 5 + K, H, W)
        :rtype:     Tensor
        """
        return torch.stack([h(inp) for h in self.heads], dim=0)
