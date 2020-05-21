cmake_minimum_required(VERSION 3.10)

# Set the project name
project(SingleImageHazeRemoval VERSION 0.0.1)

# Set required packages
set(OpenCV_DIR "/usr/share/OpenCV")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build/bin/)

# Add directories with header files
include_directories(include)

# Add main executable
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(haze_removal src/main.cpp)
target_link_libraries(haze_removal ${OpenCV_LIBS})
