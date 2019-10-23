#coding:utf-8
import numpy as np

def py_cpu_nms(class_boxes, class_box_scores, max_boxes, iou_threshold):
    """Pure Python NMS baseline."""
    x1 = class_boxes[:, 1] # x_min
    y1 = class_boxes[:, 0] # y_min
    x2 = class_boxes[:, 3] # x_max
    y2 = class_boxes[:, 2] # y_max
    # scores = class_box_scores[:]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = class_box_scores.argsort()[::-1] # 分数从大到小排序

    keep = []
    while order.size > 0 and len(keep) < max_boxes:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep