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
git submodule update --init --recursive

python -m venv .venv # Create virtual python environment
source .venv\Scripts\activate.bat # Windows
source .venv/bin/activate # Bash / Zsh / Sh

# Unoffical sdk repo
sudo apt install ros-$ROS_DISTRO-image-tools
sudo apt install ros-$ROS_DISTRO-vision-msgs
sudo apt install python3-pip clang portaudio19-dev

cd src/go2_ros2_sdk
pip install -r requirements.txt
cd ../..

source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select go2_ros2_sdk --cmake-clean-cache
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

Usage unofficial SDK:
```bash
export ROBOT_IP=192.168.1.103
export CONN_TYPE="webrtc"
ros2 launch go2_robot_sdk robot.launch.py
```