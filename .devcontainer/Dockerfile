FROM ros:humble-ros-base

# Install basic requirements as root
RUN apt update && export DEBIAN_FRONTEND=noninteractive && \
    apt install -y \
    python3-pip \
    python-is-python3 \
    curl \
    gnupg2 \
    lsb-release \
    portaudio19-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ros-humble-rviz2 \
    software-properties-common \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-map-server \
    ros-humble-nav2-util \
    ros-humble-nav2-amcl \
    ros-humble-slam-toolbox \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-twist-mux \
    ros-humble-foxglove-bridge \
    ros-humble-joy \
    ros-humble-teleop-twist-joy \
    ros-humble-compressed-image-transport \

# Install specific numpy version
RUN pip install 'numpy<2.0.0'

# Create non-root user (matches devcontainer.json)
ARG USERNAME=go2edu
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Set up workspace (owned by go2edu)
RUN mkdir -p /home/${USERNAME}/ros2_ws/src \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

# Switch to user context
USER ${USERNAME}
WORKDIR /home/${USERNAME}/ros2_ws

# Source ROS in .bashrc (user-specific)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# Environment variables (accessible to user)
ENV ROBOT_IP=""
ENV CONN_TYPE="webrtc"
ENV WEBRTC_SERVER_HOST="0.0.0.0"
ENV WEBRTC_SERVER_PORT="9991"
ENV PATH="/home/go2edu/.local/bin:${PATH}"

# Cleanup as root (then switch back)
USER root
RUN rm -rf /var/lib/apt/lists/*
USER ${USERNAME}