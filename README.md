# Human-Centered Robotics – Project 5 (KIT)

This repository contains the code and documentation for **Project #5** of the *Human-Centered Robotics* practical course at KIT.  
The goal is to develop a **safe autonomous companion walking system** using the **Unitree GO2 EDU quadruped robot**.

---

## 📌 Project Overview

The project enables the Unitree GO2 robot to:  
- **Autonomously follow** a human while maintaining a safe distance.  
- **Estimate and track human positions** in 3D using YOLO-based keypoint and bounding box tracking.  
- **Filter and stabilize detections** for reliable motion control.
- **PID control** to reach a given target position

👉 The **main components** are the `human_localizer_node` and the `camera_orientation_node`,  
which rely on the scripts in the `scripts/` folder (e.g. `human_localizer.py`, `human_tracker.py`, `keypoint_tracker.py`).  

---

## 📂 Project Structure
```bash
Human-Centered-Robotics-Project-5-KIT/
├── src/project_5_pkg/                   # Main ROS 2 package
│   ├── project_5_pkg/
│   │   ├── scripts/                     # Core tracking & localization scripts
│   │   │   ├── human_localizer.py
│   │   │   ├── human_position_filter.py
│   │   │   ├── human_tracker.py
│   │   │   └── keypoint_tracker.py
│   │   ├── bb_to_movement.py            # debugging - version without keypoint tracker
│   │   ├── camera_orientation_node.py   # Determines camera tilt
│   │   ├── human_localizer_node.py      # Main human localization node
│   │   ├── publish_pose.py              # debugging - publish human 3d position
│   │   ├── yolo_keypoint_tracker_node.py # deprecated 
│   │   └── yolo_person_tracker_node.py  # deprecated 
│   ├── config/                          # ROS 2 node configurations
│   ├── launch/                          # Launch files
│   ├── models/                          # Trained YOLO models
│   ├── test/                            # Test scripts
│   ├── setup.py / setup.cfg / package.xml
│   └── README.md
├── rosbags/                             # Recorded data for testing
├── HumanLocalizer/human_detector/       # YOLO detection + training scripts
├── cyclonedds.xml                       # DDS config for robot connection
└── README.md
``` 
---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone git@github.com:LukasInformatik/human_centered_robots_project_5.git
cd human_centered_robots_project_5
```

### 2. Initialize Submodules & Python Environment
```bash
git submodule update --init --recursive --force

python -m venv .venv
source .venv/bin/activate
```

### 3. Install Unitree SDK (Python)
```bash
cd src
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
pip install -e .
```

Additional required Python packages are listed in setup.py.
ROS dependencies are specified in package.xml.
These should be installed automatically; if not, you can find them there.

### 4. Setup Realsense ROS Wrapper
Install on the **robot (camera host)** by following the official [Intel Realsense ROS instructions](https://github.com/IntelRealSense/realsense-ros).

### 5. Setup Cyclone DDS Connection
Ensure your DDS configuration works.
You can find tutorials on how to set up your DDS connection online (e.g. [here](https://iroboteducation.github.io/create3_docs/setup/xml-config/)
Our dds config is provided in [cyclonedds.xml](cyclonedds.xml).  
If everything is set up correctly, you should see robot ros topics on your pc with:
```bash
ros2 topic list
```

### 6. Build ROS Package
```bash
# From your workspace root
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep update 
rosdep install --from-paths src/project_5_pkg --ignore-src -r -y
colcon build --packages-select project_5_pkg --cmake-clean-cache
```

## 🚀 Usage

### 1. Source Workspace
In every new terminal:
```bash
source install/setup.bash
```

### 2. Start Realsense Node (on Robot)
```bash
ros2 launch project_5_pkg realsense_local.launch
```

### 3. Start Camera Orientation Node (on PC)
Make sure DDS is running and realsense topics are visible:
```bash
ros2 run project_5_pkg camera_orientation_node
```

### 4. Start Human Localizer
```bash
ros2 run project_5_pkg human_localizer_node
```

➡️ A UI will appear where you can select the human to follow.  
➡️ The estimated 3D pose is published as TF frames and can be visualized in **RViz**.

---

