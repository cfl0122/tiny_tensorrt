#!/usr/bin/env python2


from __future__ import print_function

import numpy as np
import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
from PIL import ImageDraw

# from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
import common,cv2




def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.shape[1], np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.shape[0], np.floor(y_coord + height + 0.5).astype(int))

        image_raw = cv2.rectangle(image_raw,(left, top), (right, bottom), (0,0,255))
        image_raw = cv2.putText(image_raw,'{0} {1:.2f}'.format(all_categories[category], score),(left, top - 12),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255))

    return image_raw

def get_engine(TRT_LOGGER, onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def init():

    global inputs, outputs, bindings, stream,engine,TRT_LOGGER,context
    TRT_LOGGER = trt.Logger()
    onnx_file_path = 'yolov3.onnx'
    engine_file_path = "yolov3.trt"
    engine = get_engine(TRT_LOGGER, onnx_file_path, engine_file_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

def myinfer(image,context, inputs, outputs, bindings, stream):
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (416, 416) 
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(image)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_HW = image_raw.shape[:2]
    H,W = shape_orig_HW

    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26)]
    # Do inference with TensorRT
    
    trt_outputs = []

        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    inputs[0].host = image
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.2,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_HW))
    # print(boxes,classes,scores)
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    if boxes is not None:
        obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
        output_image_path = 'dog_bboxes.png'
        cv2.imshow("test",obj_detected_img)
    if boxes is not None:
        boxes[:,0] = boxes[:,0]/W
        boxes[:,1] = boxes[:,1]/H
        boxes[:,2] = boxes[:,2]/W
        boxes[:,3] = boxes[:,3]/H
    return boxes,classes,scores    

    # cv2.imwrite(output_image_path, obj_detected_img)
    # print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

    

if __name__ == '__main__':
    init()
    image =cv2.imread('dog.jpg')    
    myinfer(image)
