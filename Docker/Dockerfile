FROM amd64/ros:noetic-perception-focal


ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=noetic
ARG NPROC=4


WORKDIR /root


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y \
    wget unzip zip tar vim curl \
    libboost-all-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libeigen3-dev \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev libceres-dev\
    libvtk7-dev libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module 

RUN apt-get update && apt-get install -y libatlas-base-dev gfortran \
    libparmetis-dev \
    libtbb-dev \
    software-properties-common \
    git \
    gdb \
    build-essential \
    cmake \
    python3-pip \
    libopencv-contrib-dev \
    libopencv-dev \
    libpcl-dev && \
    rm -rf /var/lib/apt/lists/*


# install ros packages
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-perception && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-vcstool python3-catkin-tools python3-osrf-pycommon


RUN rosdep update --rosdistro $ROS_DISTRO

RUN apt-get install -y \
    ros-${ROS_DISTRO}-hector-trajectory-server \
    ros-${ROS_DISTRO}-rviz \
    ros-${ROS_DISTRO}-image-geometry ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-cv-bridge pkg-config autoconf

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc


WORKDIR /root
