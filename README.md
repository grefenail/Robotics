

````markdown
# 🤖 Ground Human-Following Robot with Recovery System

This repository contains the code and documentation for my **Capstone Project 2** at Sunway University (BSc Computer Science).  
It implements a **ROS-based ground robot** that can detect, track, and follow a designated human target — and recover their location if temporarily lost.

📄 **Full Report:** [FR_21007364_Sep23.pdf](FR_21007364_Sep23.pdf)  
📊 **Activity Log:** [AL_21007364_Sep23.pdf](AL_21007364_Sep23.pdf)

---

## ✨ Features
- **Real-time Human Detection** using **SSD MobileNet V2** with TensorFlow.
- **Main Target Identification** via **HSV color segmentation**.
- **Depth & Position Tracking** using Intel RealSense RGB-D camera.
- **ROS Publish/Subscribe** system to share target data between nodes.
- **Recovery Module** that estimates the target's last known position using trigonometric calculations.
- **Navigation Control** for the Reeman Big Dog Chassis.

---

## 🛠 Hardware & Software Requirements
- **Hardware**
  - Reeman Big Dog Chassis with LIDAR
  - Intel RealSense D435 (or compatible RGB-D camera)
  - Laptop running Ubuntu 16.04 (ROS compatible)

- **Software**
  - ROS (Robot Operating System)
  - Python 3.x  
    ```bash
    pip install pyrealsense2 numpy opencv-python tensorflow==1.15
    sudo apt install ros-<distro>-cv-bridge ros-<distro>-image-transport
    ```

---

## 📂 Project Structure
````

├── human\_detection.py     # Detects humans, IDs main target, publishes pose & depth
├── hd2.py                 # Subscribes to target data, calculates position, sends nav commands
├── pose.py                # Utility for pose retrieval
├── clr.py / hdlr.py       # Support modules (color filtering, lidar processing)
├── AL\_21007364\_Sep23.pdf  # Activity log
├── FR\_21007364\_Sep23.pdf  # Final report

````

---

## ⚙️ Setup & Running
1. **Start ROS core**
   ```bash
   roscore
````

2. **Run the human detection node**

   ```bash
   python human_detection.py
   ```
3. **Run the navigation/recovery node**

   ```bash
   python hd2.py
   ```

---

## 🧠 How It Works

1. **Detection** – SSD model finds all humans in the camera frame.
2. **Target Selection** – HSV color segmentation chooses the main target.
3. **Tracking** – Depth & bounding box center offset guide the robot's rotation and movement.
4. **Recovery** – If target is lost, last known orientation & depth are used to calculate a predicted location.
5. **Navigation** – Coordinates are sent to `/cmd/nav` API for the robot to move toward the target.

---

## 📸 System Diagram

![System Diagram](docs/system_diagram.png)

---

## 📅 Development Timeline

See the **Capstone 2 Gantt Charts** in [AL\_21007364\_Sep23.pdf](AL_21007364_Sep23.pdf) for the full development schedule.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

```

---

If you want, I can **also extract a real diagram** from your Capstone PDF and save it as `docs/system_diagram.png` so it displays perfectly in the README.  
Do you want me to grab one of your diagrams and include it? That would make the page look much more professional.
```
