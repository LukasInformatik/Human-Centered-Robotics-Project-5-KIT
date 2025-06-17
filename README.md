# Praktikum Human Centered Robots

This repository contains the code and documentation for Project #5 of the Human-Centered Robotics practical course at KIT. The goal is to develop a safe autonomous companion walking system using the Unitree GO2 EDU quadruped robot.

### Project overview

The project focuses on enabling the Unitree GO2 robot to:

- Follow a human companion autonomously while maintaining a safe distance.
- Avoid obstacles dynamically in real-time.
- React to human gestures or voice commands (optional extension).
- Operate safely in human-populated environments.

## Installation

### Prerequisites
1. Docker ([Installation Guide](https://www.docker.com/))
    - Required for containerized development and deployment.
2. VS Code ([Download](https://code.visualstudio.com/))
   - For development with the Dev Containers extension.
3. X Server (for GUI applications in Docker)
   - Linux: Install xorg and enable X11 forwarding. This should already be the case on most linux version.
   - Windows: Use VcXsrv and WSL (guide below).
   - macOS: Use XQuartz (guide below).

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

## Usage

Devcontainers are Docker containers configured to provide a complete, isolated development environment. VS Code integrates with Docker to automatically load a dev container when you open a project.

For this open VS code and install the extension [**Dev Containers**](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). When opening this the root folder of this project, VS code will automatically prompt to open the workspace in a devcontainer. 

If this does not happen just press `CTRL + SHIFT + p` or the `><` button in the bottom left and type `open` and select `Dev Containers: Reopen in Container`. This takes a while when building the first time ... go get a coffee :)

### TODO
Here goes some more documentation on how to launch ROS programs. We will need to figure out when you have to source ros configs and how to launch stuff. The devcontainer should already run `colon build` but depending on how we set up our own package this documentation goes here.

Some problems can arrise due to Docker not having permissions to communicate with the X Server. On Mac and Linux running `xhost +local:docker` can help.


## Additional guides

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