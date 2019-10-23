#!/usr/bin/env python
#coding=utf-8

import rospy
import pylab as plt
import sys
import os,time
import cv2
import cv2 as cv
import tensorrt as trt
import common
import threading

from threading import Lock

from onnx_to_tensorrt_4 import get_engine,myinfer

from sensor_msgs.msg import Image, CameraInfo

from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from detector.msg import BoundingBox,BoundingBoxArray
from std_msgs.msg import Header



class Detection():

    def __init__(self):

        self.node_name = "cv_bridge_demo"
        self.header = Header(frame_id='base')
        self.boundingbox = BoundingBox()
        self.boundingboxarray = BoundingBoxArray()

        self._frame_lock = Lock()   
        self.show_frame_lock = Lock()     
        cv2.namedWindow('test')
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.cleanup)

        # 创建 rgb图像 显示窗口

        self.cv_window_name = self.node_name



        self.TRT_LOGGER = trt.Logger()
        dirpath = os.path.dirname(__file__)
        self.onnx_file_path = os.path.join(dirpath,'yolov3.onnx')
        self.engine_file_path = os.path.join(dirpath,"yolov3.trt")
        self.engine = get_engine(self.TRT_LOGGER, self.onnx_file_path, self.engine_file_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)



        # 创建 ros 图 到 opencv图像转换 对象

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)

        self.detection_pub = rospy.Publisher("/detection_result", BoundingBoxArray, queue_size=100)



        # 登陆信息

        rospy.loginfo("Waiting for image topics...")

        rospy.wait_for_message("/usb_cam/image_raw", Image)

        rospy.loginfo("Ready.")



    # 收到rgb图像后的回调函数

    def image_callback(self, ros_image):

        # 转换图像格式到opencv格式

        try:

            image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        except CvBridgeError as e:

            print(e)

        # 转换成 numpy 图像数组格式
        frame = np.array(image, dtype=np.uint8)
        if self._frame_lock.acquire():
            self.frame = frame
            self._frame_lock.release()


        # 检测键盘按键事件
        self.keystroke = cv2.waitKey(5)

        if self.keystroke != -1:

            cc = chr(self.keystroke & 255).lower()

            if cc == 'q':

                # The user has press the q key, so exit

                rospy.signal_shutdown("User hit q key to quit.")

    def infer(self):
            self.header.seq = self.header.seq + 1
            self.header.stamp = rospy.Time.now()
            self.boundingboxarray.header = self.header
            self.boundingboxarray.data = []
            frame = self.frame
            boxes,classes,scores,show_frame = myinfer(frame, self.context, self.inputs, self.outputs, self.bindings, self.stream)
            if self.show_frame_lock.acquire():
                self.show_frame = show_frame
                self.show_frame_lock.release()
            if classes is not None:
                # self.obj_detected_img = obj_detected_img
                for i in range(len(classes)):
                    pass
                    self.boundingbox.x1 = boxes[i][0]
                    self.boundingbox.y1 = boxes[i][1]
                    self.boundingbox.w = boxes[i][2]
                    self.boundingbox.h = boxes[i][3]
                    self.boundingbox.categories = classes[i]
                    self.boundingbox.confidences = scores[i]
                    # print(self.boundingboxarray)
                    self.boundingboxarray.data.append(self.boundingbox)

            self.detection_pub.publish(self.boundingboxarray)
            # return fimg
    def show(self):
        self.show_frame = None
        while True:
                print('/////////////////////////')                
		if self.show_frame is not None:
		    cv2.imshow('test',self.show_frame)
	        cv2.waitKey(50)

    def startShow(self):
        print('****************************************')
        if not hasattr(self,'thread') or not self.thread.isAlive():
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.thread =threading.Thread(target=self.show)
            self.thread.setDaemon(True)
            self.thread.start()

    def cleanup(self):

        print("Shutting down vision node.")

        cv2.destroyAllWindows()   


def init():

    global inputs, outputs, bindings, stream,engine,TRT_LOGGER,context
    TRT_LOGGER = trt.Logger()
    onnx_file_path = 'yolov3.onnx'
    engine_file_path = "yolov3.trt"
    engine = get_engine(TRT_LOGGER, onnx_file_path, engine_file_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# 主函数    

def main(args):       

    try:
        obj_detected_img = None
        # cv2.namedWindow('test',flags=cv2.WINDOW_NORMAL)
        detection = Detection()
        detection.startShow()
        # rate = rospy.Rate(10)
        #rospy.spin()
        # plt.ion()
        # fimg = None
        while not rospy.is_shutdown():
            detection.infer()
            # if obj_detected_img is not None:
                # cv2.imshow('test2',detection.obj_detected_img)
                # cv2.waitkey(20)
            # if fimg is not None:
            #    plt.imshow(fimg)
            #    plt.pause(0.01)
            # plt.show()
            # rate.sleep()          
            # time.sleep(0.01)


    except KeyboardInterrupt:

        print("Shutting down vision node.")

        # cv.DestroyAllWindows()
if __name__=='__main__':
    main("detect")
