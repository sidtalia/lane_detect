#!/usr/bin/env python3
# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import tf.transformations

import test
from test import *
# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError
import traceback
import time
import math as m
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
plt.axis('equal')

# Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
rospy.init_node('opencv_example', anonymous=True)


# Initialize the CvBridge class
bridge = CvBridge()

DEBUG = True

class lane_driver:
  def __init__(self):
    self.pos = np.zeros(3)  # using ENU ref frame according to ROS REP-105
    self.init_pos = np.zeros(3)
    self.init = False
    self.lane_init = False
    self.vel = np.zeros(3)
    self.velBF = np.zeros(3)
    self.rotBF = np.zeros(3)
    self.quat = np.array([1,0,0,0])
    self.rpy = np.zeros(3)
    self.LWB2 = 1.5
    self.pc = np.zeros(3)  # center curve polynomial in local ref frame

    self.L_list = []
    self.R_list = []
    self.C_list = []

    self.ox = 512/2
    self.oy = 512/2
    self.fov_x = 104
    self.fov_y = 72
    self.height = 1
    DEG2RAD = 1/57.3
    self.K_v = m.tan(self.fov_y*DEG2RAD/2)
    self.K_h = m.tan(self.fov_x*DEG2RAD/2)

    self.min_dist = 4  # minimum measurable distance from camera
    self.lookahead = 5  # lookahead distance upto which lane markers are considered

    self.x = None
    self.y = None
    # Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
    sub_image = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
    sub_odom = rospy.Subscriber("/odom/pixhawk", Odometry, self.odom_callback)

  def quaternion_to_angle(self, q):
    """
    Convert a quaternion _message_ into an angle in radians.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return np.array([roll, pitch, yaw])

  def odom_callback(self, msg):
    self.pos[1] = msg.pose.pose.position.x
    self.pos[0] = msg.pose.pose.position.y
    self.pos[2] = msg.pose.pose.position.z
    self.pos -= self.init_pos

    if(self.init == False):
      self.init_pos = self.pos
      self.init = True
      return

    self.vel[0] = msg.twist.twist.linear.x
    self.vel[1] = msg.twist.twist.linear.y
    self.vel[2] = msg.twist.twist.linear.z

    self.rotBF[0] = msg.twist.twist.angular.x
    self.rotBF[1] = msg.twist.twist.angular.y
    self.rotBF[2] = msg.twist.twist.angular.z

    self.quat[0] = msg.pose.pose.orientation.x
    self.quat[1] = msg.pose.pose.orientation.y
    self.quat[2] = msg.pose.pose.orientation.z
    self.quat[3] = msg.pose.pose.orientation.w

    self.rpy = self.quaternion_to_angle(msg.pose.pose.orientation)

  def img2XY(self, X_pix, Y_pix,height,K_v,K_h,Xmax,Ymax,X_Center,Y_Center_default,pitch):
    Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
    Y = height*Ymax/(K_v*(Y_pix - Y_Center))
    X = Y*K_h*(X_pix - X_Center)/Xmax
    return X,Y

  def project_cam_to_bodyframe(self, x, y):
    x = x[0]
    y = y[0]

    lane_x = []
    lane_y = []
    for i in range(len(x)):
      xl, yl = self.img2XY(np.array(x[i]), np.array(y[i]) + self.oy, self.height, self.K_v, self.K_h,
                      self.ox*2, self.oy*2, self.ox, self.oy, -0.08 - self.rpy[1])
      index = np.where(np.fabs(yl) < 15)
      xl = -xl[index]
      yl = yl[index]
      lane_x.append(xl)
      lane_y.append(yl)
    return lane_y, lane_x
            

  # def transform_body_to_worldframe(self, x, y):

  # def transform_world_to_bodyframe(self, x, y, radius = 10):


  # def lane_stuff(self, x, y):
  #   ## initialze the centerline with the assumption that points are good and pre-aligned (mostly)
  #   xb, yb = self.project_cam_to_bodyframe(x, y)

  #   left_index = 0
  #   right_index = 1
  #   if(xb[0][i] < xb[1][i]):
  #     left_index = 1
  #     right_index = 0

  #   if(not self.lane_init):
  #     self.lane_init = True
  #     ## curve fitting and then sampling would probably work better
  #     self.LWB2 = 0.5*(yb[left_index][0] - yb[right_index][0])  # yes I know this does not generalize shutup
  #     ycb = np.array(list(np.array(yb[left_index]) - self.LWB2) + list(np.array(yb[right_index]) + self.LWB2))
  #     xcb = np.array(list(xb[left_index]) + list(xb[right_index]))  # lists can be added like this 
  #     pc = np.poly1d(np.polyfit(xcb, ycb, 2))
  #     xcb = np.arange(self.min_dist, self.min_dist + self.lookahead, 0.5)
  #     ycb = np.poly1d(xcb)  # get corresponding points
  #     xcw, ycw = self.transform_body_to_worldframe(xcb, ycb)
  #     self.xcw_list = np.array(list(self.xcw_list) + list(xcw))
  #     self.ycw_list = np.array(list(self.ycw_list) + list(ycw))
  #     return

  #   xcb, ycb = self.transform_world_to_bodyframe(self.xcw_list, self.ycw_list, radius = 10)
  #   C = np.polyfit(xcb, ycb, 2)
  #   pc = np.poly1d(C)
  #   end_tangent = 2*C[0]*xcb[-1] + C[1]
  #   theta = np.arctan2(1, end_tangent)  # atan(dx/dy)
  #   shift_vec = self.LWB2*np.array([np.cos(theta), np.sin(theta)])

  #   xb = np.array(xb)
  #   yb = np.array(yb)
  #   for i in range(2):
  #     limit = np.max(np.where(xb[i] < xcb[-1] + 3))
  #     start = np.min(np.where(xb[i] > xcb[-1]))
  #     xb[i] = xb[i][start:limit]
  #     yb[i] = yb[i][start:limit]

  #   xb[left_index] -= shift_vec[0]
  #   yb[left_index] -= shift_vec[1]

  #   xb[right_index] += shift_vec[0]
  #   xb[right_index] += shift_vec[1]

  #   xcb = list(xcb) + list(xb[left_index]) + list(xb[right_index])
  #   ycb = list(ycb) + list(yb[left_index]) + list(yb[right_index])
  #   # update polynomial i guess
  #   xcb, ycb = np.array(xcb), np.array(ycb)
  #   pc = np.poly1d(np.polyfit(xcb, ycb, 2))
  #   xcb = np.arange(self.min_dist, self.min_dist + self.lookahead, 0.5)
  #   ycb = np.poly1d(xcb)  # get corresponding points
  #   xcw, ycw = self.transform_body_to_worldframe(xcb, ycb)
  #   self.xcw_list = np.array(list(self.xcw_list) + list(xcw))
  #   self.ycw_list = np.array(list(self.ycw_list) + list(ycw))


  # Define a callback for the Image message
  def image_callback(self, img_msg):
    global DEBUG
    # Try to convert the ROS Image message to a CV2 Image
    try:
      cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")# [:,:,:3]
      cv_image = cv_image[cv_image.shape[0]//2:,:,:]
      # img_path2=r'/home/iitdautomation/DLive/PINet/dataset/Test_images/11A00148.JPG'
      # cv_image = cv2.imread(img_path2)
      now = time.time()
      x, y, lane_img = run_lane_detect(cv_image)
      x, y = self.project_cam_to_bodyframe(x, y)
      self.x = x
      self.y = y
      if DEBUG:
        cv2.imshow("test", cv_image)
        cv2.waitKey(1)
        dt = time.time() - now
        # print(1/dt)

    except Exception:
      print(traceback.format_exc())

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
lane_detector = lane_driver()  # create object

while not rospy.is_shutdown():
  try:
    if(lane_detector.x != None):
      # plt.clf()
      for i in range(len(lane_detector.x)):
        plt.scatter(np.array(lane_detector.x[i]),np.array(lane_detector.y[i]))
      plt.show()
      plt.pause(0.01)
  except:
    pass
