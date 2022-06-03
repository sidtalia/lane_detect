#!/usr/bin/env python3
# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from nav_msgs.msg import Odometry
import tf.transformations
from scipy.ndimage import gaussian_filter 
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
    self.done_processing = True
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
    self.LWB2 = 2.0
    self.pc = np.zeros(3)  # center curve polynomial in local ref frame
    self.posX = 0
    self.posY = 0

    self.xcw_list = []
    self.ycw_list = []

    # TODO: this information should be obtained from the camera info topic
    self.ox = 512/2
    self.oy = 512/2 + 64
    self.fov_x = 90
    self.fov_y = 72
    self.height = 1.2
    self.pitch = 0
    DEG2RAD = 1/57.3
    self.K_v = m.tan(self.fov_y*DEG2RAD/2)
    self.K_h = m.tan(self.fov_x*DEG2RAD/2)

    self.min_dist = 4  # minimum measurable distance from camera
    self.lookahead = 15  # lookahead distance upto which lane markers are considered

    self.x = None
    self.y = None
    self.start_lat = 0
    self.start_lon = 0
    self.have_gps = False
    # Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
    sub_image = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback, queue_size=1)
    sub_pose = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback, queue_size=1)
    sub_gps = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback, queue_size=1)
    self.lane_pub = rospy.Publisher("/lane_node/points", PoseArray, queue_size = 10)
    self.path_pub = rospy.Publisher("/lane_node/path", PoseArray, queue_size = 1)

    self.last_inference = time.time()
    self.expected_frame_time = 0.05
    self.reject_counter = 0
    self.frame_counter = 0
    while not self.have_gps:
      time.sleep(0.05)
    self.gps_x, self.gps_y = find_route(self.start_lat, self.start_lon, 28.545925, 77.179456, self.start_lat, self.start_lon)  # 28.545925, 77.179456
    self.gps_xb = 0
    self.gps_yb = 0
    self.gps_shift = np.zeros(2)
    self.shift = np.zeros(2)

  def quaternion_to_angle(self, q):
    """
    Convert a quaternion _message_ into an angle in radians.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return np.array([roll, pitch, yaw])

  def gps_callback(self, msg):
    if(not self.have_gps):
      self.have_gps = True
      self.start_lat = msg.latitude
      self.start_lon = msg.longitude
      self.posX, self.posY = 0,0
    else:
      self.posX, self.posY = calcposNED(msg.latitude, msg.longitude, self.start_lat, self.start_lon)

  def pose_callback(self, msg):
    self.pos[0] = msg.pose.position.x
    self.pos[1] = msg.pose.position.y
    self.pos[2] = msg.pose.position.z

    if(self.init == False):
      self.init_pos = np.copy(self.pos)
      self.init = True

    self.pos -= self.init_pos

    self.quat[0] = msg.pose.orientation.x
    self.quat[1] = msg.pose.orientation.y
    self.quat[2] = msg.pose.orientation.z
    self.quat[3] = msg.pose.orientation.w

    rpy = self.quaternion_to_angle(msg.pose.orientation)
    rpy_diff = rpy - self.rpy
    self.rpy = rpy
    self.inferred_yaw = self.rpy[2] - 2/57.3 
    self.pitch = 0.8*(rpy_diff[1] + self.pitch)

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
                      self.ox*2, self.oy*2, self.ox, self.oy, 0.01 - self.pitch) # todo: fix this self.oy should not be added directly to Y
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
                    self.ox*2, self.oy*2, self.ox, self.oy, 0.01 - self.pitch, 512)
    return xl, yl - self.oy + 64
            
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
    if(len(left) < 3 or len(right) < 3):
      return
    else:
      self.LWB2 = self.LWB2*0.9 + 0.1*width

  def interpolate(self, X, Y):
    for i in range(3):
      Y = np.dstack((Y[:-1],Y[:-1] + np.diff(Y)/2.0)).ravel()
      Y = np.dstack((Y[:-1],Y[:-1] + np.diff(Y)/2.0)).ravel()
      X = np.dstack((X[:-1],X[:-1] + np.diff(X)/2.0)).ravel()
      X = np.dstack((X[:-1],X[:-1] + np.diff(X)/2.0)).ravel()

      Y = gaussian_filter(Y,sigma=1)
      X = gaussian_filter(X,sigma=1)
    return X, Y

  def publish_lane_markers(self, xb, yb, left_index, right_index, xw, yw, th):
    try:
      left_points_y = np.copy(yb[left_index])
      left_points_x = np.copy(xb[left_index])
      right_points_y = np.copy(yb[right_index])
      right_points_x = np.copy(xb[right_index])
      lxw, lyw = self.transform_body_to_worldframe(left_points_x, left_points_y, xw, yw, th)
      rxw, ryw = self.transform_body_to_worldframe(right_points_x, right_points_y, xw, yw, th)

      lxw += self.init_pos[0]
      rxw += self.init_pos[0]
      lyw += self.init_pos[1]
      ryw += self.init_pos[1]

      ps = PoseArray()
      ps.header.frame_id = "/odom"
      ps.header.stamp = rospy.Time.now()

      for i in range(len(lxw)):
        pose = Pose()
        pose.position.x = lxw[i]
        pose.position.y = lyw[i]
        ps.poses.append(pose)
      for i in range(len(rxw)):
        pose = Pose()
        pose.position.x = rxw[i]
        pose.position.y = ryw[i]
        ps.poses.append(pose)
      self.lane_pub.publish(ps)
    except:
      pass
    return

  def publish_gps_path(self, xw, yw, th):
      path = PoseArray()
      path.header.frame_id = "/odom"
      path.header.stamp = rospy.Time.now()

      x, y = self.transform_body_to_worldframe(np.copy(self.gps_xb), np.copy(self.gps_yb), xw, yw, th)
      x += self.init_pos[0]
      y += self.init_pos[1]
      dy = np.diff(y)
      dx = np.diff(x)
      heading = np.arctan2(dy, dx)  # ENU ref frame
      for i in range(len(self.gps_xb)-1):
        pose = Pose()
        pose.position.x = x[i]
        pose.position.y = y[i]
        pose.position.z = heading[i]
        path.poses.append(pose)
      self.path_pub.publish(path)


  def lane_stuff(self, x, y, xw, yw, th):
    ## initialze the centerline with the assumption that points are good and pre-aligned (mostly)
    xb, yb = self.project_cam_to_bodyframe(x, y)

    gps_xb, gps_yb = self.transform_world_to_bodyframe(np.copy(self.gps_x + self.gps_shift[0]), np.copy(self.gps_y + self.gps_shift[1]), xw, yw, th, radius = 5)
    index = np.where((gps_xb > 0) & (gps_xb < 20))
    gps_xb = gps_xb[index]
    gps_yb = gps_yb[index]
    self.gps_xb, self.gps_yb = gps_xb, gps_yb
    self.publish_gps_path(xw, yw, th)
    left_index = 0
    right_index = 1
    try:
      if(yb[0].mean() < yb[1].mean()):
        left_index = 1
        right_index = 0
    except:
      return
    
    self.find_lane_width(np.copy(yb[left_index]), np.copy(yb[right_index]))  # update lane width using lane markers

    self.publish_lane_markers(xb, yb, left_index, right_index, xw, yw, th)

    left_points = yb[left_index] - self.LWB2
    right_points = yb[right_index] + self.LWB2

    ycb = np.concatenate((left_points, right_points))
    xcb = np.concatenate((xb[left_index], xb[right_index]))
    if(len(ycb) < 5):
      self.reject_counter += 1
      return
    pc = np.poly1d(np.polyfit(xcb, ycb, 2))
    max_dist = np.max(xcb)
    self.x, self.y = xcb, ycb

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
    index = np.where((last_x > 5) & (last_x < self.lookahead))

    last_x = last_x[index]
    last_y = last_y[index]


    separation =  np.fabs(last_y.mean() - ycb.mean())
    num_points = 10.0/min(max(1.0, separation),5.0)
    interp_dist = max_dist/num_points

    xcb = np.arange(self.min_dist, max_dist, interp_dist)
    ycb = pc(xcb)  # get corresponding points

    x_filt = np.concatenate((last_x, self.x))
    y_filt = np.concatenate((last_y, self.y))
    index = np.where( (x_filt < 20) & (x_filt > 0) )
    x_filt = x_filt[index]
    y_filt = y_filt[index]

    pc_filt = np.poly1d(np.polyfit(x_filt, y_filt, 2))
    xcb = np.arange(self.min_dist, max_dist, 0.5)
    ycb = pc_filt(xcb)

    yc_gps = pc_filt(gps_xb)
    shift_lateral = y_filt.mean() - gps_yb.mean()
    detected_angle = m.atan2(y_filt[-1] - y_filt[0],x_filt[-1] - x_filt[0])
    predicted_angle = m.atan2(gps_yb[-1] - gps_yb[0], gps_xb[-1] - gps_xb[0])
    delta = detected_angle - predicted_angle
    weight = m.cos(delta*2)**2
    shift = np.zeros(2)
    shift[0], shift[1] = self.rotate_body_to_worldframe(0, shift_lateral, th)
    if(len(right_points) > 5):
      self.gps_shift[0] += weight*shift[0]*min(1, 0.05*len(right_points))
      self.gps_shift[1] += weight*shift[1]*min(1, 0.05*len(right_points))
      # self.gps_shift[0] = max(min(self.gps_shift[0],5),-5)
      # self.gps_shift[1] = max(min(self.gps_shift[1],5),-5)

    xcw, ycw = self.transform_body_to_worldframe(self.x, self.y, xw, yw, th)
    self.xcw_list = np.concatenate((self.xcw_list, xcw))
    self.ycw_list = np.concatenate((self.ycw_list, ycw))
    self.last_pos = np.array([xw, yw])

  # Define a callback for the Image message
  def image_callback(self, img_msg):
    if(time.time() - self.last_inference < self.expected_frame_time or not self.done_processing or not self.have_gps):
      return
    self.done_processing = False
    self.last_inference = time.time()
    world_X, world_Y, world_th = self.posX, self.posY, self.inferred_yaw
    # Try to convert the ROS Image message to a CV2 Image
    global DEBUG
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
        x, y = self.project_bodyframe_to_cam(np.copy(self.gps_xb), np.copy(self.gps_yb))
        pts = np.column_stack((np.int32(x),np.int32(y)))
        cv2.polylines(lane_img, [pts], False, (255,255,255), 5)
      except:
        print(traceback.format_exc())
      if DEBUG:
        lane_img = cv2.resize(lane_img, (400, 400))
        cv2.imshow("test", lane_img)
        cv2.waitKey(1)

    except Exception:
      print(traceback.format_exc())
    self.done_processing = True

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
lane_detector = lane_driver()  # create object

while not rospy.is_shutdown():
  time.sleep(0.1)

