# Praktikum Human Centered Robots

This repository contains the code and documentation for Project #5 of the Human-Centered Robotics practical course at KIT. The goal is to develop a safe autonomous companion walking system using the Unitree GO2 EDU quadruped robot.

### Project overview

The project focuses on enabling the Unitree GO2 robot to:

- Follow a human companion autonomously while maintaining a safe distance.
- Avoid obstacles dynamically in real-time.
- React to human gestures or voice commands (optional extension).
- Operate safely in human-populated environments.

### Setup instructions

1. Clone the repository
```bash
git clone git@github.com:LukasInformatik/human_centered_robots_project_5.git
cd human_centered_robots_project_5
```
2. Open in devcontainer as described in Usage

**Local setup:**
```bash
source /opt/ros/humble/setup.bash 
 git clone --recurse-submodules https://github.com/abizovnuralem/go2_ros2_sdk.git src/go2_sdk  
 pip install -r src/go2_sdk/requirements.txt 
 sudo apt-get update  
 sudo apt-get upgrade -y 
 rosdep update 
 rosdep install --from-paths src --ignore-src -r -y 
 sudo apt install ros-humble-librealsense2*
 colcon build --symlink-install
```

## Additional guides

A deployment option via devcontainer is on the equivalent named branch. 

**Windows guide**

Open PowerShell as Administrator and run:
```Powershell
wsl --install
wsl --set-default-version 2 # Set WSL 2 as default
```
Enable WSL 2 integration in Docker settings:
- Open Docker Desktop
- Go to Settings > Resources > WSL Integration
- Enable integration with your installed distro

Download and install VcXsrv from [sourceforge.net](https://sourceforge.net/projects/vcxsrv/).
Run XLaunch (from Start menu) and configure:
- Select "Multiple windows"
- Display number: -1 (auto-select)
- Start no client
- Check "Disable access control" (for simplicity)
- Save configuration for future use

**MacOS guide**

Install XQuartz:
- Download XQuartz from [www.xquartz.org](https://www.xquartz.org/).
- Run the installer and follow the prompts
- Restart your Mac to complete installation

Configure XQuartz:
- Open XQuartz (from Applications > Utilities)
- Go to XQuartz > Preferences:
  - In the "Security" tab: Check "Allow connections from network clients"
  - In the "Output" tab: Uncheck "Enable syncing" (can improve performance)