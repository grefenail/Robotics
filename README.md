
````markdown
# Ground Human-Following Robot with Recovery System

This project is my Capstone Project 2 for the Bachelor of Computer Science program at Sunway University.  
It is a ROS-based ground robot that can detect, track, and follow a specific human target, and recover their position if temporarily lost.

## Project Summary
The system uses an Intel RealSense RGB-D camera and a TensorFlow SSD MobileNet V2 model to detect humans in real time.  
A color-based identification method (HSV filtering) is applied to distinguish the main target from other people.  
Depth data from the camera is used to measure the distance to the target, while the center offset of the bounding box is used to guide robot rotation.

If the main target is lost for more than 5 seconds, the system records the last observed orientation and depth, then calculates the target’s estimated location using trigonometric functions.  
The predicted coordinates are sent to the robot’s navigation API so it can move toward the last known position.

## Main Features
- Human detection using SSD MobileNet V2.
- Target identification using HSV color segmentation.
- Depth-based distance measurement.
- ROS publish/subscribe communication between detection and navigation modules.
- Recovery module that predicts the target’s location when lost.
- Navigation control for the Reeman Big Dog robot.

## How It Works
1. **Detection** – The camera feed is processed by the SSD model to detect humans.
2. **Target Selection** – The main target is identified using HSV color filtering.
3. **Tracking** – Depth and position offset data are used to control rotation and movement.
4. **Recovery** – If the target is lost, last known position and depth are used to predict their new location.
5. **Navigation** – The robot moves toward the predicted location using its navigation API.

## Files
- `human_detection.py` – Detects humans, identifies the main target, and publishes target location data.
- `hd2.py` – Subscribes to target data, calculates estimated position, and sends navigation commands.
- `pose.py`, `clr.py`, `hdlr.py` – Support modules.
- `AL_21007364_Sep23.pdf` – Activity log.
- `FR_21007364_Sep23.pdf` – Final project report.

## Requirements
- Reeman Big Dog robot with LIDAR.
- Intel RealSense D435 or compatible RGB-D camera.
- ROS on Ubuntu 16.04.
- Python 3.x with:
  - pyrealsense2
  - numpy
  - opencv-python
  - tensorflow 1.x
  - rospy and ROS Python packages.

## Running the Project
1. Start the ROS core:
   ```bash
   roscore
````

2. Run the human detection script:

   ```bash
   python human_detection.py
   ```
3. Run the navigation/recovery script:

   ```bash
   python hd2.py
   ```


