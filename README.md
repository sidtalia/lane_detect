# lane_detect

Author: Sidharth Talia, Navneeth K P (IIT-Delhi DLive project)

This repository provides a ros-compatible lane estimation system. The repository includes code from the original [PINET](https://github.com/koyeongmin/PINet) repository. The purpose of this system is to estimate the lane center using detections from the PINET lane detector as well as estimates based on the car's global position and routing information. The routing information is obtained using pyroutelib3 and the global position is obtained from an GPS inertial navigation system. In our case, we are using a pixhawk 2.4.8 running ardupilot 4.2 with a ublox F9P GPS.

## Brief overview:
The lane detection points from PINET are de-warped from the camera perspective into the body frame coordinate system around the car (X:Front Y:Left Z:Up). These points are then used to obtain a center line. Using the routing library and the global position data, road center line can be obtained in the car's reference frame. Due to GPS errors, this lane center estimate can be incorrect, and thus there would be some lateral error in the detected lane center and the estimated lane center. The detected lane center is used to correct the offset in the estimated lane center over time. The benefit of this approach is that once the lateral error has been corrected, lane position can be estimated even when there are no detections simply by utilizing the routing data and the global position. The global position comes from an inertial navigation system and is relatively noise free. 

The estimated lane center is published as the global path to be followed. One may consider this system analogous to the global planner used by indoor robots. The system also publishes the lane marker points on a separate topics in case the local planner needs them.

## Dependencies/Requirements:
This package depends on the following software components:
1) ROS (tested with noetic)
2) Python3
3) [Dependencies of PINET](https://github.com/koyeongmin/PINet#dependency)
4) mavros ([install instructions](https://ardupilot.org/dev/docs/ros-install.html))
5) pyroutelib3 (pip install pyroutelib3)

## Remarks:
The system as it exists has been hardcoded to work with the zed2 camera and the ardupilot system.

## Usage:
```
roscore
```

In a new tab:
```
rosrun lane_detect lane_node_gps.py
```

## Note:
This system is under active development
