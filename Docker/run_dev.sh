#!/bin/bash

PROJECT_NAME="point_cloud_projection"
IMAGE_NAME="${PROJECT_NAME}:noetic-ros"
DATA_PATH="/media/${USER}/zhipeng_usb/datasets"
DATA2_PATH="/media/${USER}/zhipeng_8t/datasets"



docker build -t $IMAGE_NAME -f "${HOME}/vscode_projects/direct_visual_lidar_calibration/${PROJECT_NAME}/catkin_ws/src/${PROJECT_NAME}/Docker/Dockerfile" .


xhost +local:root

docker run \
    -e DISPLAY=$DISPLAY \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${HOME}/vscode_projects/direct_visual_lidar_calibration/${PROJECT_NAME}/catkin_ws:/root/catkin_ws \
    -v ${DATA_PATH}:/root/data1 \
    -v ${DATA2_PATH}:/root/data2 \
    --privileged \
    --cap-add sys_ptrace \
    --runtime=nvidia \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# docker run --rm -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# xhost -local:root