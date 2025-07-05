cmake_minimum_required(VERSION 3.3)
project(point_cloud_projection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


option(ENABLE_CMAKE_DEBUG "Enable detailed debug messages" ON)

if(ENABLE_CMAKE_DEBUG)
    # Debug or Release
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Wall -Wuninitialized -fno-omit-frame-pointer")



find_package(PCL 1.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(VTK REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    rospy
    std_msgs
    sensor_msgs 
    cv_bridge
    message_filters
)

catkin_package(
    CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs 
    cv_bridge
    message_filters
)

include(${VTK_USE_FILE})
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}

)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ColoriseScan
    src/ColoriseScan.cc
    src/utils.cc
)

target_link_libraries(ColoriseScan
    PUBLIC
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    ${catkin_LIBRARIES}
)


