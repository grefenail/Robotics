Paste this into your README.md and edit small details like file paths or demo images.

# 🤖 Ground Human-Following Robot with Recovery System

A ROS-based autonomous ground robot that can **detect, track, and follow a human target**, and **recover** when the target is lost using predicted coordinates.

![Robot Demo](docs/demo.gif) <!-- (Optional: add a short demo gif or image) -->

---

## 🧩 Project Overview
This project was developed as **Capstone Project 2** for the *Bachelor of Computer Science (Hons)* program at **Sunway University**.  
It demonstrates the integration of **computer vision**, **sensor fusion**, and **robot navigation** using **ROS** and **TensorFlow**.

---

## ⚙️ System Architecture


<img width="1077" height="603" alt="image" src="https://github.com/user-attachments/assets/47e66662-6594-47ab-b390-52f515f65395" />


---

## 🧠 Main Features
- Real-time **human detection and tracking** using TensorFlow SSD MobileNet V2  
- **HSV color filtering** to isolate the primary target from background noise  
- **Recovery mode** that predicts target location when temporarily lost  
- Integration with **ROS navigation stack** for autonomous motion  
- Depth-based distance estimation and orientation correction  

---

## 🧪 Technologies Used
| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming | Python, ROS |
| Vision | TensorFlow, OpenCV |
| Sensors | Intel RealSense RGB-D |
| Simulation | Webots |
| ML Model | SSD MobileNet V2 |
| Control | Kalman Prediction, PID |

---

## 🧰 Installation & Setup
1. Clone this repository  
   ```bash
   git clone https://github.com/grefenail/Robotics.git
   cd Robotics/ground_following_robot
   

2. Run the human detection script:

   ```bash
   python human_detection.py
   ```
3. Run the navigation/recovery script:

   ```bash
   python hd2.py
   ```


