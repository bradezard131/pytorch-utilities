import torch
from torchvision import ops


def get_positives(gt_boxes, gt_classes, rois, roi_classes, iou_thresh=0.5):
    iou_matrix = ops.box_iou(gt_boxes, rois) > iou_thresh
    class_matrix = gt_classes.view(-1, 1) == roi_classes.view(1, -1)
    pos_matrix = iou_matrix & class_matrix
    return pos_matrix.any(dim=1)


def mean_average_precision(gt_boxes, gt_classes, rois, roi_classes, iou_thresh=0.5):
    positives = get_positives(gt_boxes, gt_classes, rois, roi_classes, iou_thresh)
    return positives.float().sum() / float(positives.size(0))


def class_average_precision(gt_boxes, gt_classes, rois, roi_classes, num_classes, iou_thresh=0.5):
    average_precisions = torch.zeros((num_classes,), dtype=torch.long)
    positives = get_positives(gt_boxes, gt_classes, rois, roi_classes, iou_thresh)
    targets = gt_classes.unique()
    class_grouping = gt_classes.view(1,-1) == targets.view(-1,1)
    class_positives = positives & class_grouping
    average_precisions[targets] = class_positives.float().sum(dim=1) / class_grouping.float().sum(dim=1)
    return average_precisions
