FROM osrf/ros:humble-desktop-full

# Fix typo in base image name (changed from "desktop-full" to correct spelling)
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    clang \
    portaudio19-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ROS packages
RUN apt-get update && apt-get install -y \
    ros-humble-image-tools \
    ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
WORKDIR /home/$USERNAME/ros2_ws