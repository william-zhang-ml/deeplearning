from typing import Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torchvision.ops.boxes import box_iou


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


def assign_to_prior(boxes: Iterable[Iterable[float]],
                    priors: Iterable[Iterable[float]]) -> Tensor:
    """ Matches bounding box to best-fit prior box.

    :param boxes:  bounding boxes (width, height) to match, (N, 2)
    :type  boxes:  Iterable[float]
    :param priors: prior boxes to choose from (width, height), (P, 2)
    :type  priors: Iterable[float]
    :return:       indices to matched prior box (N, )
    :rtype:        Tensor
    """
    boxes = torch.tensor(boxes, dtype=torch.float32)    # cast
    priors = torch.tensor(priors, dtype=torch.float32)  # cast

    # center all boxes around the origin
    boxes = torch.stack([
        -boxes[:, 0] / 2,
        -boxes[:, 1] / 2,
        boxes[:, 0] / 2,
        boxes[:, 1] / 2], dim=1)
    priors = torch.stack([
        -priors[:, 0] / 2,
        -priors[:, 1] / 2,
        priors[:, 0] / 2,
        priors[:, 1] / 2], dim=1)

    iou = box_iou(boxes, priors)  # (N, P)
    return iou.argmax(dim=-1)     # match by IOU


def plot_prior_regions(priors: Iterable[Iterable[float]],
                       w_min: float,
                       w_max: float,
                       h_min: float,
                       h_max: float,
                       n_grid: int = 100,
                       figsize: Tuple[float, float] = (8, 4.5)):
    """ Make a plot showing prior boxes will claim bounding boxes.

    :param priors:  prior boxes (width, height), (P, 2)
    :type  priors:  Iterable[Iterable[float]]
    :param w_min:   minimum bounding box width to consider
    :type  w_min:   float
    :param w_max:   maximum bounding box width to consider
    :type  w_max:   float
    :param h_min:   minimum bounding box height to consider
    :type  h_min:   float
    :param h_max:   maximum bounding box height to consider
    :type  h_max:   float
    :param n_grid:  num of width and height points considered, defaults to 100
    :type  n_grid:  int, optional
    :param figsize: figure width and height in inches
    :type: figsize: Tuple[float, float]
    :return:       figure and axes
    :rtype:        -
    """
    priors = np.array(priors)  # cast

    # define a grid of bounding boxes
    wm, hm = np.meshgrid(
        np.linspace(w_min, w_max, n_grid),
        np.linspace(h_min, h_max, n_grid))
    wm, hm = wm.reshape(n_grid ** 2), hm.reshape(n_grid ** 2)
    boxes = np.stack([wm, hm], axis=-1)

    # assign bounding boxes to priors
    assign = assign_to_prior(boxes, priors)

    # plot and annotate
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(priors[:, 0], priors[:, 1], 'wo', zorder=10)
    for idx in range(len(priors)):
        mask = assign == idx
        ax.plot(boxes[mask, 0], boxes[mask, 1], '.', markersize=1)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_xlim([w_min, w_max])
    ax.set_ylim([h_min, h_max])
    fig.tight_layout()
    return fig, ax


def append_yolo_targets(annot: DataFrame,
                        bin_w: float,
                        bin_h: float,
                        priors: DataFrame) -> DataFrame:
    """ Compute and append Yolo target values to annotation DataFrame (in-place).

    :param annot:  ground truth bounding boxes (columns 'xc', 'yc', 'w', 'h')
    :type  annot:  DataFrame
    :param bin_w:  prediction cell width
    :type  bin_w:  float
    :param bin_h:  prediction cell height
    :type  bin_h:  float
    :param priors: prior boxes (columns 'w', 'h')
    :type  priors: DataFrame
    :return:       annotations w/appended target values
                   (columns 'prior', 'ix', 'tx', 'iy', 'ty', 'tw', 'th')
    :rtype:        DataFrame
    """
    annot['prior'] = assign_to_prior(
        annot[['w', 'h']].values,
        priors[['w', 'h']].values)
    annot['ix'], x_rem = np.divmod(annot.xc.values, bin_w)
    annot['tx'] = x_rem / bin_w
    annot['iy'], y_rem = np.divmod(annot.yc.values, bin_h)
    annot['ty'] = y_rem / bin_h
    annot['tw'] = np.log(annot.w.values / priors.w.iloc[annot.prior].values)
    annot['th'] = np.log(annot.h.values / priors.h.iloc[annot.prior].values)
    return annot
