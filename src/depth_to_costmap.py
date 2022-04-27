#!/usr/bin/env python3
import numpy as np
import math as m
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import time

rospy.init_node('depth2Cost', anonymous=True)


# Initialize the CvBridge class
bridge = CvBridge()

class depth_to_costmap:
  def __init__(self):
    self.fx = 1
    self.fy = 1
    self.cx = 1
    self.cy = 1
    self.cam_info_init = False
    sub_image = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.image_callback, queue_size=1)
    sub_info = rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, self.info_callback, queue_size=1)
    self.max_height = 0.8
    self.min_height = -1.2
    self.resolution = 0.5
    self.mapX = 10
    self.mapY = 10  # 10 x 10 grid with the car's camera sitting at (mapX/2, 0)
    self.originX = 0
    self.originY = self.mapY/2
    self.ignore_dist = 1.5
    self.max_dist = 10
    self.grid_X, self.grid_Y = int(self.mapX/self.resolution), int(self.mapY/self.resolution)

  def info_callback(self, msg):
    if(self.cam_info_init):
      return
    self.cam_info_init = True
    self.fx = msg.K[0]
    self.fy = msg.K[4]
    self.cx = msg.K[2]
    self.cy = msg.K[5]

  def image_callback(self, img_msg):
    if(not self.cam_info_init):
      return
    depth_image = bridge.imgmsg_to_cv2(img_msg)
    dim_U, dim_V = depth_image.shape[1], depth_image.shape[0]  # dim 0 is height, dim 1 is width (X)
    
    now = time.time()
    occupancy_grid = np.zeros((self.grid_X, self.grid_Y), dtype="uint8")
    for i in range(dim_V):
      for j in range(dim_U):
        X = depth_image[i][j]
        Y = ((self.cx - j)/self.fx) * X
        Z = ((self.cy - i)/self.fy) * X
        
        ## point inside the map grid?
        if(Z < self.max_height and Z > self.min_height and X > self.ignore_dist and X < self.mapX and m.fabs(Y) < self.mapY/2):
          trunc_X = int(X/self.resolution)
          trunc_Y = int((Y + self.originY)/self.resolution)
          occupancy_grid[trunc_Y][trunc_X] = 255
    print(time.time() - now)

d2C = depth_to_costmap()
while not rospy.is_shutdown():
  time.sleep(0.1)