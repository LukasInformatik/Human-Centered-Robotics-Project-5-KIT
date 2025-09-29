# Praktikum Human Centered Robots

This repository contains the code and documentation for Project #5 of the Human-Centered Robotics practical course at KIT. The goal is to develop a safe autonomous companion walking system using the Unitree GO2 EDU quadruped robot.

### Project overview

The project focuses on enabling the Unitree GO2 robot to:

- Follow a human companion autonomously while maintaining a safe distance.
- Avoid obstacles dynamically in real-time.
- React to human gestures or voice commands (optional extension).
- Operate safely in human-populated environments.

### Setup instructions

Clone the repository
```bash
git clone git@github.com:LukasInformatik/human_centered_robots_project_5.git
cd human_centered_robots_project_5
```

Local setup:
```bash
git submodule update --init --recursive --force

python -m venv .venv # Create virtual python environment
source .venv\Scripts\activate.bat # Windows
source .venv/bin/activate # Bash / Zsh / Sh

cd src/unitree_sdk2_python
pip install -e .
cd ../..
```

For realsense ros wrapper, follow instructions on their [github page](https://github.com/IntelRealSense/realsense-ros).

Afterward, build project_5 ros package:
```bash
# cd to ws directory
source /opt/ros/$ROS_DOSTRO/setup.bash
rosdep update 
rosdep install --from-paths src/project_5_pkg --ignore-src -r -y
colcon build --packages-select project_5_pkg --cmake-clean-cache
```

### Usage

When opening a new terminal allways source ws setup
```bash
source install/setup.bash
```

Start realsense node with configuration (this should be done on the robot)
```bash
ros2 launch project_5_pkg realsense_local.launch
```

Make sure your DDS connection to the robot stands. An example configuration XML file can be found [here](cyclonedds.xml).
The ROS topics of the extension board should now be visible:
```bash
ros2 topic list
```

Start yolo node on PC
```bash
ros2 run project_5_pkg yolo_person_tracker_node.py
```
