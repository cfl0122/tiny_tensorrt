#coding:utf-8
import numpy as np
from my_nms import py_cpu_nms

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''得到相对于原图正确的边界框

    args:
        box_xy: shape为(?, 13, 13, 3, 2)
        box_wh: shape为(?, 13, 13, 3, 2)
        input_shape: [416, 416]
        image_shape: 原始输入图片的尺寸

    returns:
        boxes: shape为(?, 13, 13, 3, 4)，相对于原图的边界框
    '''
    box_yx = box_xy[..., ::-1]

    box_hw = box_wh[..., ::-1]
    input_shape = input_shape.astype(box_yx.dtype)
    image_shape = image_shape.astype(box_yx.dtype)
    # new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    # offset = (input_shape - new_shape) / 2. / input_shape
    # scale = input_shape / new_shape
    # box_yx = (box_yx - offset) * scale
    # box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1]*image_shape[0],  # y_min
        box_mins[..., 1:2]*image_shape[1],  # x_min
        box_maxes[..., 0:1]*image_shape[0],  # y_max
        box_maxes[..., 1:2] *image_shape[1] # x_max
    ], axis=-1)

    # boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes  # boxes的shape为(?, 13, 13, 3, 4)

def yolo_head(feats, anchors, num_classes, input_shape):
    '''从最后的卷积特征图中得到边界框的参数

    args:
        feats: 为输入的特征图，比如(?, 13, 13, 255)这个特征图
        anchors: 为其对应的anchor box大小[[116.  90.], [156. 198.], [373. 326.]]
        num_classes: 为80个类
        input_shape: [416, 416]

    returns:
        box_xy: shape为(?, 13, 13, 3, 2)
        box_wh: shape为(?, 13, 13, 3, 2)
        box_confidence: (?, 13, 13, 3, 1)
        box_class_probs: (?, 13, 13, 3, 80)
    '''
    num_anchors = len(anchors)  # 为3， 比如[[116.  90.], [156. 198.], [373. 326.]]
    # anchors_tensor [[[[[116.  90.] [156. 198.] [373. 326.]]]]]
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = np.shape(feats)[1:3]  # height, width # 比如(13, 13)
    # grid_y与grid_x的shape都是(13, 13, 1, 1)
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)  # 一定要加上axis=-1
    grid.astype(feats.dtype)

    # feats的shape为(?, 13, 13, 3, 85)
    feats = np.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 每个空间网格点和锚定大小的调整预处理。
    # box_xy的shape为(?, 13, 13, 3, 2)
    box_xy = (sigmoid(feats[..., :2]) + grid) / grid_shape[::-1]
    # box_wh的shape为(?, 13, 13, 3, 2)
    box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / input_shape[::-1]
    # box_confidence为(?, 13, 13, 3, 1)
    box_confidence = sigmoid(feats[..., 4:5])
    # box_class_probs为(?, 13, 13, 3, 80)
    box_class_probs = sigmoid(feats[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''处理卷积层的输出，也就是处理特征图，以下以feats为(?, 13, 13, 255)说明

    args:
        feats: 为输入的特征图，比如(?, 13, 13, 255)这个特征图
        anchors: 为其对应的anchor box大小[[116.  90.], [156. 198.], [373. 326.]]
        num_classes: 为80个类
        input_shape: [416, 416]
        image_shape: 为原图尺寸

    returns:
        boxes: shape为(? * 13 * 13 * 3, 4)，其中的参数为相对于输入的原图的边界框坐标
        box_scores: shape为(?, 13, 13, 3, 80)，代表每个框的每个类别的自信度
    '''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    # 得到的boxes为相对于原图的boxes，boxes的shape为(?, 13, 13, 3, 4)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # boxes的shape为(? * 13 * 13 * 3, 4)
    boxes = np.reshape(boxes, [-1, 4])
    # box_scores的shape为(?, 13, 13, 3, 80)，代表每个框的每个类别的自信度
    box_scores = box_confidence * box_class_probs
    # box_scores的shape为(? * 13 * 13 * 3, 80)
    box_scores = np.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs, anchors, num_classes, image_shape,
              max_boxes=20, score_threshold=.6, iou_threshold=.2):
    '''在给定输入的情况下评估Yolo模型，返回过滤之后的boxes

    args:
        yolo_outputs: 为一个list，里面有三个元素，每个元素对应着三个不同的特征图的输出，每个元素的shape为(?, ?, ?, 255), 13或者26或者52，80个类为255
        anchors: 为numpy矩阵，shape为(9, 2)，分别对应9个anchor box的大小，前三个对应52*52的特征图，最后三个对应13*13的特征图
        num_classes: 为识别的类别数，对于coco数据来说为80
        input_image_shape: 为原始输入图片的大小
        score_threshold: 为自信度阈值
        iou_threshold: nms的阈值

    returns:
        boxes_: shape为(经过score过滤和nms之后留下来的边框个数, 4) (可能会有重复的边框被留下，但是是不同类别的)
        scores_: shape为(经过score过滤和nms之后留下来的边框个数, ) 记录下框的自信度
        classes_: shape为(经过score过滤和nms之后留下来的边框个数, ) 记录下框的类别索引
    '''
    num_layers = len(yolo_outputs)  # 为3
    # anchor_mask为[[6,7,8], [3,4,5], [0,1,2]]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # yolo_outputs中第一个元素的shape为(?, 13, 13, 255)， 第二个为(?, 26, 26, 255)，第三个为(?, 52, 52, 255)
    input_shape = np.array(np.shape(yolo_outputs[0])[1:3]) * 32  # input_shape为[416, 416]
    boxes = []
    box_scores = []
    for l in range(num_layers):  # 循环不同的特征图
        # 输出的_boxes的shape为(? * 13 * 13 * 3, 4), 是相对于原始输入图片的框大小
        # _box_scores的shape为(? * 13 * 13 * 3, 80), 为每个框的自信度
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # boxes的shape为(? * (13 * 13 + 26 * 26 + 52 * 52) * 3, 4)
    boxes = np.concatenate(boxes, axis=0)
    # box_scores的shape为(? * (13 * 13 + 26 * 26 + 52 * 52) * 3, 80)
    box_scores = np.concatenate(box_scores, axis=0)

    # mask的shape为(? * (13 * 13 + 26 * 26 + 52 * 52) * 3, 80)，其中的元素为True或者False
    mask = box_scores >= score_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # class_boxes的shape为(mask中为True的框的个数, 4)，记录的是每个留下来的框的box
        class_boxes = boxes[mask[:, c]]
        # class_box_scores的shape为(mask中为True的框的个数, )，记录的是每个留下来的框的分数
        class_box_scores = box_scores[:, c][mask[:, c]]
        # nms_index的shape应该是(经过NMS留下来的boxes的个数, ), 其中里面记录的是经过NMS留下来的boxes的索引
        nms_index = py_cpu_nms(class_boxes, class_box_scores, max_boxes, iou_threshold=iou_threshold)
        # class_boxes的shape为(经过NMS留下来的boxes的个数, 4)
        class_boxes = class_boxes[nms_index]
        # class_box_scores的shape为(经过NMS留下来的boxes的个数, )，里面记录的是经过score筛选和NMS留下的每个框的分数
        class_box_scores = class_box_scores[nms_index]
        # classes的shape为(经过NMS留下来的boxes的个数, )，记录下框的类别索引
        classes = np.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # 得到所有类别的
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
