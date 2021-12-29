"""
Distance/complete intersection-over-union (D/CIoU) implementations.

Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression
https://arxiv.org/pdf/1911.08287.pdf
"""
import torch
from torch import Tensor
from torchvision.ops import box_iou


PI = torch.acos(torch.tensor(-1)).item()


def diou_penalty(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    """ Computes centerpoint distance penalty for each bounding box pair.

    :param bbox1: xyxy-format bounding boxes (N, 4)
    :type  bbox1: Tensor
    :param bbox2: corresponding xyxy-format bounding boxes (N, 4)
    :type  bbox2: Tensor
    :return:      centerpoint distance penalty, eqn 6 in paper (N, 1)
    :rtype:       Tensor
    """
    # compute squared distance between centerpoints
    center1 = (bbox1[:, :2] + bbox1[:, 2:]) / 2
    center2 = (bbox2[:, :2] + bbox2[:, 2:]) / 2
    center_sq_dist = (center1 - center2).square().sum(dim=1, keepdim=True)

    # compute squared diagonal length of enclosing bounding box
    enclos_lo = torch.min(bbox1[:, :2], bbox2[:, :2])
    enclos_hi = torch.max(bbox1[:, 2:], bbox2[:, 2:])
    enclos_sq_len = (enclos_hi - enclos_lo).square().sum(dim=1, keepdim=True)

    # eqn 6
    return center_sq_dist / enclos_sq_len


def aspect_penalty(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    """ Computes aspect ratio penalty for each bounding box pair.

    :param bbox1: xyxy-format bounding boxes (N, 4)
    :type  bbox1: Tensor
    :param bbox2: corresponding xyxy-format bounding boxes (N, 4)
    :type  bbox2: Tensor
    :return:      aspect ratio penalty, eqn 9 in paper (N, 1)
    :rtype:       Tensor
    """
    # compute width divided by height
    aspect1 = (bbox1[:, 2] - bbox1[:, 0]) / (bbox1[:, 3] - bbox1[:, 1])
    aspect2 = (bbox2[:, 2] - bbox2[:, 0]) / (bbox2[:, 3] - bbox2[:, 1])

    # un-normalized square "error"
    raw_penalty = (torch.atan(aspect1) - torch.atan(aspect2)) ** 2

    # eqn 9
    return 4 / (PI ** 2) * raw_penalty.unsqueeze(1)


def diou_loss(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    """ Computes distance intersection-over-union for each bounding box pair.

    :param bbox1: xyxy-format bounding boxes (N, 4)
    :type  bbox1: Tensor
    :param bbox2: corresponding xyxy-format bounding boxes (N, 4)
    :type  bbox2: Tensor
    :return:      distance intersection-over-union, eqn 7 in paper (N, 1)
    :rtype:       Tensor
    """
    # compute vanilla iou
    bbox_pairs = zip(bbox1, bbox2)
    iou = torch.cat(
        [box_iou(a.unsqueeze(0), b.unsqueeze(0)) for a, b in bbox_pairs],
        dim=0)

    # eqn 7
    return 1 - iou + diou_penalty(bbox1, bbox2)


def ciou_loss(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    """ Computes complete intersection-over-union for each bounding box pair.

    :param bbox1: xyxy-format bounding boxes (N, 4)
    :type  bbox1: Tensor
    :param bbox2: corresponding xyxy-format bounding boxes (N, 4)
    :type  bbox2: Tensor
    :return:      complete intersection-over-union, eqn 10 in paper (N, 1)
    :rtype:       Tensor
    """
    # compute vanilla IoU
    bbox_pairs = zip(bbox1.unsqueeze(1), bbox2.unsqueeze(1))
    iou = torch.cat([box_iou(a, b) for a, b in bbox_pairs], dim=0)

    # weigh aspect ratio penalty more or less based on IoU, eqn 11
    v = aspect_penalty(bbox1, bbox2)

    # eqn 10
    return 1 - iou + diou_penalty(bbox1, bbox2) + v * v / (1 - iou + v + 1e-9)
