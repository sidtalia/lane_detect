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
import gps_lane
from gps_lane import *
plt.ion()
fig = plt.figure()
plt.axis('equal')

# Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
rospy.init_node('opencv_example', anonymous=True)


# Initialize the CvBridge class
bridge = CvBridge()

DEBUG = True

def claheFilter(img):
    assert img.dtype == np.uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(img)


def hlsClahe(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(img)
    h = claheFilter(h)
    out = cv2.merge([h, l, s])
    img = cv2.cvtColor(out, cv2.COLOR_HLS2RGB)
    return img


def rgbclahe(img):
    r, g, b = cv2.split(img)
    r = claheFilter(r)
    g = claheFilter(g)
    b = claheFilter(b)
    out = cv2.merge([r, g, b])
    return out

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
    self.inferred_yaw = 0
    self.LWB2 = 2
    self.pc = np.zeros(3)  # center curve polynomial in local ref frame
    self.posX = 0
    self.posY = 0

    self.xcw_list = []
    self.ycw_list = []

    self.ox = 512/2
    self.oy = 512/2 + 64
    self.fov_x = 104
    self.fov_y = 72
    self.height = 1.25
    DEG2RAD = 1/57.3
    self.K_v = m.tan(self.fov_y*DEG2RAD/2)
    self.K_h = m.tan(self.fov_x*DEG2RAD/2)

    self.min_dist = 4  # minimum measurable distance from camera
    self.lookahead = 15  # lookahead distance upto which lane markers are considered

    self.x = None
    self.y = None
    # Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
    sub_image = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback, queue_size=1)
    sub_odom = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback, queue_size = 1)
    self.last_inference = time.time()
    self.expected_frame_time = 0.1
    self.reject_counter = 0
    self.frame_counter = 0
    self.gps_x, self.gps_y = find_route(28.547373, 77.183662,28.544931, 77.194203, 28.544728, 77.183394)
    self.gps_xb = 0
    self.gps_yb = 0
    self.shift = np.zeros(2)
    self.done_processing = True

  def quaternion_to_angle(self, q):
    """
    Convert a quaternion _message_ into an angle in radians.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return np.array([roll, pitch, yaw])

  def odom_callback(self, msg):
    self.pos[0] = msg.pose.pose.position.x
    self.pos[1] = msg.pose.pose.position.y
    self.pos[2] = msg.pose.pose.position.z

    if(self.init == False):
      self.init_pos = np.copy(self.pos)
      self.init = True

    self.pos -= self.init_pos
    self.vel[0] = msg.twist.twist.linear.x
    self.vel[1] = msg.twist.twist.linear.y
    self.vel[2] = msg.twist.twist.linear.z

    self.rotBF[0] = msg.twist.twist.angular.x
    self.rotBF[1] = msg.twist.twist.angular.y
    self.rotBF[2] = -msg.twist.twist.angular.z

    self.quat[0] = msg.pose.pose.orientation.x
    self.quat[1] = msg.pose.pose.orientation.y
    self.quat[2] = msg.pose.pose.orientation.z
    self.quat[3] = msg.pose.pose.orientation.w

    self.rpy = self.quaternion_to_angle(msg.pose.pose.orientation)
    # print(self.rpy*57.3)
    self.inferred_yaw = self.rpy[2] + 2/57.3 #self.inferred_yaw + self.rotBF[2]*0.04 ## 50 hz
    # R = np.array([[m.cos(self.inferred_yaw), -m.sin(self.inferred_yaw)],
    #               [m.sin(self.inferred_yaw), m.cos(self.inferred_yaw)]])
    # x, y = np.matmul(R,np.array([np.linalg.norm(self.vel)*0.02, 0]))
    self.posX, self.posY = self.pos[0], self.pos[1] #self.posX + x, self.posY + y

  def img2XY(self, X_pix, Y_pix,height,K_v,K_h,Xmax,Ymax,X_Center,Y_Center_default,pitch):
    Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
    Y = height*Ymax/(K_v*(Y_pix - Y_Center))
    X = Y*K_h*(X_pix - X_Center)/Xmax
    return X,Y

  def XY2img(self,X,Y,height,K_v,K_h,Xmax,Ymax,X_Center,Y_Center_default,pitch,cutoff):
    Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
    Y_pix = (Y_Center + height*Ymax/(Y*K_v))
    X_pix = (X_Center + X*Xmax/(Y*K_h))
    if(cutoff<Ymax):
      index = np.where((Y_pix>0)&(Y_pix<cutoff))
    else: 
      index = np.where((Y_pix>0)&(Y_pix<Ymax))
    Y_pix = Y_pix[index]
    X_pix = X_pix[index]
    return X_pix,Y_pix

  def project_cam_to_bodyframe(self, x, y):
    x = x[0]
    y = y[0]

    lane_x = []
    lane_y = []
    for i in range(len(x)):
      xl, yl = self.img2XY(np.array(x[i]), np.array(y[i]) + self.oy - 64, self.height, self.K_v, self.K_h,
                      self.ox*2, self.oy*2, self.ox, self.oy, 0.01) # todo: fix this self.oy should not be added directly to Y
      index = np.where(yl < self.min_dist + self.lookahead)
      xl = -xl[index]
      yl = yl[index]
      lane_x.append(xl)
      lane_y.append(yl)
    return lane_y, lane_x

  def project_bodyframe_to_cam(self, x, y):
    x_ = -y
    y_ = x

    xl, yl = self.XY2img(x_, y_, self.height, self.K_v, self.K_h,
                    self.ox*2, self.oy*2, self.ox, self.oy, -0.08, 512)
    return xl, yl - self.oy
            
  ## I'm performing a 2D transformation from body to world frame. 
  ## This transform will hold so long as we travel on a surface that is "mostly flat"
  def transform_body_to_worldframe(self, x, y, wx, wy, th):
    R = np.zeros((2,2))
    ct, st = m.cos(th), m.sin(th)
    R[0,0], R[0,1], R[1,0], R[1,1] = ct, -st, st, ct
    X = np.array(x)
    Y = np.array(y)
    if(len(X)==0):
      return
    V = np.array([X,Y])
    V_ = np.matmul(R, V)
    X = V_[0,:] + wx
    Y = V_[1,:] + wy
    return X, Y

  def rotate_body_to_worldframe(self, x, y, th):
    R = np.zeros((2,2))
    ct, st = m.cos(th), m.sin(th)
    R[0,0], R[0,1], R[1,0], R[1,1] = ct, -st, st, ct
    X = np.array([x,y])
    V = np.matmul(R, X)
    return V[0], V[1]

  def transform_world_to_bodyframe(self, x, y, xw, yw, th, radius=5):
    # print(x[:2],xw,y[:2],yw)
    x -= xw
    y -= yw
    R = np.zeros((2,2))
    ct, st = m.cos(-th), m.sin(-th)
    R[0,0], R[0,1], R[1,0], R[1,1] = ct, -st, st, ct
    X = np.array(x)
    Y = np.array(y)
    V = np.array([X,Y])
    O = np.matmul(R, V)
    x, y = O[0,:], O[1,:]
    
    return x, y

  def point_culling(self, last_points, xw, yw):
    if(len(self.xcw_list) < last_points):
      return
    self.xcw_list = self.xcw_list[-last_points:]
    self.ycw_list = self.ycw_list[-last_points:]
    pts = np.column_stack((self.xcw_list - xw, self.ycw_list -yw))
    pts = np.array(pts)
    index = np.where(np.linalg.norm(pts) < 15)
    self.xcw_list = self.xcw_list[index]
    self.ycw_list = self.ycw_list[index]

  def find_lane_width(self, left, right):
    width = 0.5*np.fabs(left.mean() - right.mean())
    if(len(left) < 10 or len(right) < 10):
      return
    else:
      self.LWB2 = self.LWB2*0.9 + 0.1*width

  def lane_stuff(self, x, y, xw, yw, th):
    ## initialze the centerline with the assumption that points are good and pre-aligned (mostly)
    xb, yb = self.project_cam_to_bodyframe(x, y)

    left_index = 0
    right_index = 1
    try:
      if(yb[0].mean() < yb[1].mean()):
        left_index = 1
        right_index = 0
    except:
      return
    
    self.find_lane_width(np.copy(yb[left_index]), np.copy(yb[right_index]))  # update lane width using lane markers

    left_points = yb[left_index] - self.LWB2
    right_points = yb[right_index] + self.LWB2
    if(np.fabs(left_points.mean() - right_points.mean()) > 0.5):
      if(np.fabs(left_points.mean()) < np.fabs(right_points.mean())):
        use_index = left_index
        ycb = left_points
        xcb = xb[left_index]
      else:
        use_index = right_index
        ycb = right_points
        xcb = xb[right_index]
    else:
      ycb = np.concatenate((left_points, right_points))
      xcb = np.concatenate((xb[left_index], xb[right_index]))
    if(np.fabs(ycb[:len(ycb)//2].mean()) > self.LWB2 or len(ycb) < 5):
      # print("hit")
      self.reject_counter += 1
      # print(100*self.reject_counter/self.frame_counter)
      return
    pc = np.poly1d(np.polyfit(xcb, ycb, 2))
    max_dist = np.max(xcb)
    
    if(not self.lane_init):
      xcb = np.arange(self.min_dist, max_dist, 0.5)
      ycb = pc(xcb)  # get corresponding points
      xcw, ycw = self.transform_body_to_worldframe(xcb, ycb, xw, yw, th)
      self.xcw_list = xcw
      self.ycw_list = ycw
      self.lane_init = True
      self.last_pos = np.array([xw, yw])
      return

    last_x, last_y = self.transform_world_to_bodyframe(np.copy(self.xcw_list), np.copy(self.ycw_list), xw, yw, th, radius = 5)
    index = np.where((last_x > 5) & (last_x < 20))
    # if(len(index) < 20):
    #   # self.reject_counter += 1
    #   index = np.where(last_x > 0)
    #   # return
    last_x = last_x[index]
    last_y = last_y[index]
    index = np.where(np.fabs(last_y) < self.LWB2)
    last_x = last_x[index]
    last_y = last_y[index]

    # last_x = last_x[-40:]
    # last_y = last_y[-40:]  # limit number of points

    separation = np.fabs(last_y.mean() - ycb.mean())
    num_points = 50.0/min(max(1.0, separation),5.0)
    interp_dist = max_dist/num_points

    xcb = np.arange(self.min_dist, max_dist, interp_dist)
    ycb = pc(xcb)  # get corresponding points

    x_filt = np.concatenate((last_x, xcb))
    y_filt = np.concatenate((last_y, ycb))
    pc_filt = np.poly1d(np.polyfit(x_filt, y_filt, 2))
    xcb = np.arange(self.min_dist, max_dist, 0.5)
    ycb = pc_filt(xcb)

    gps_xb, gps_yb = self.transform_world_to_bodyframe(np.copy(self.gps_x - self.init_pos[0]), np.copy(self.gps_y - self.init_pos[1]), xw, yw, th, radius = 5)
    index = np.where((gps_xb > 5) & (gps_xb < 20))
    gps_xb = gps_xb[index]
    gps_yb = gps_yb[index]
    yc_gps = pc_filt(gps_xb)
    shift_lateral = yc_gps.mean() - gps_yb.mean()
    self.gps_xb, self.gps_yb = gps_xb, gps_yb
    shift = np.zeros(2)
    shift[0], shift[1] = self.rotate_body_to_worldframe(0, shift_lateral, th)
    if(len(left_points) > 10 or len(right_points) > 10):
      self.gps_x += shift[0]*0.1
      self.gps_y += shift[1]*0.1

    xcw, ycw = self.transform_body_to_worldframe(xcb, ycb, xw, yw, th)
    self.x, self.y = xcb, ycb
    self.xcw_list = np.concatenate((self.xcw_list, xcw))
    self.ycw_list = np.concatenate((self.ycw_list, ycw))
    self.last_pos = np.array([xw, yw])

  # Define a callback for the Image message
  def image_callback(self, img_msg):
    if(time.time() - self.last_inference < self.expected_frame_time or (not self.done_processing) ):
      return
    self.done_processing = False
    self.last_inference = time.time()
    global DEBUG
    world_X, world_Y, world_th = self.posX, self.posY, self.inferred_yaw
    # Try to convert the ROS Image message to a CV2 Image
    try:
      cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")# [:,:,:3]
      cv_image = cv_image[cv_image.shape[0]//2:,:,:]
      now = time.time()
      self.frame_counter += 1
      x, y, lane_img = run_lane_detect(cv_image)
      dt = time.time() - now
      # print(1/dt)
      self.lane_stuff(x, y, world_X, world_Y, world_th)
      # self.x, self.y = self.project_cam_to_bodyframe(x, y)
      try:
        last_x, last_y = self.transform_world_to_bodyframe(np.copy(self.xcw_list), np.copy(self.ycw_list), world_X, world_Y, world_th, radius = 5)
        index = np.where(last_x > self.min_dist)
        last_x = last_x[index]
        last_y = last_y[index]
        index = np.where(np.fabs(last_y) < self.LWB2)
        last_x = last_x[index]
        last_y = last_y[index]

        last_x = last_x[-40:]
        last_y = last_y[-40:]  # limit number of points      
        pc_filt = np.poly1d(np.polyfit(last_x, last_y, 2))
        max_dist = np.max(last_x)
        fit_x = np.arange(self.min_dist, max_dist, 0.5)
        fit_y = pc_filt(fit_x)
        x, y = self.project_bodyframe_to_cam(np.copy(fit_x), np.copy(fit_y))
        pts = np.column_stack((np.int32(x),np.int32(y)))
        cv2.polylines(lane_img, [pts], False, (0,0,0), 5)
        x, y = self.project_bodyframe_to_cam(np.copy(self.gps_xb), np.copy(self.gps_yb))
        pts = np.column_stack((np.int32(x),np.int32(y)))
        cv2.polylines(lane_img, [pts], False, (255,255,255), 5)
        self.fit_x, self.fit_y = self.transform_body_to_worldframe(fit_x, fit_y, world_X, world_Y, world_th)
      except:
        print(traceback.format_exc())
      if DEBUG:
        cv2.imshow("test", lane_img)
        cv2.waitKey(1)

    except Exception:
      print(traceback.format_exc())
    self.done_processing = True

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
lane_detector = lane_driver()  # create object

while not rospy.is_shutdown():
  time.sleep(0.1)
  try:
    # if(lane_detector.x.any()):
    plt.clf()
    # for i in range(len(lane_detector.x)):
    # plt.scatter(lane_detector.xcw_list,lane_detector.ycw_list)
    # for i in range(len(lane_detector.x)):
    plt.scatter(lane_detector.gps_x - lane_detector.init_pos[0], lane_detector.gps_y - lane_detector.init_pos[1], color="black")
    plt.scatter(lane_detector.xcw_list, lane_detector.ycw_list, color="blue")
    plt.scatter(lane_detector.pos[0], lane_detector.pos[1], color="red")
    plt.scatter(lane_detector.fit_x, lane_detector.fit_y, color="green")    
    plt.axis("equal")
    plt.show()
    plt.pause(0.01)
  except:
    pass
    # print(traceback.format_exc())
