cmake_minimum_required(VERSION 3.1)
project(BoxesMatchProject)

set(CMAKE_CXX_FLAGS "-fPIC") #

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the pybind11 and Eigen subdirectories
# Assuming pybind11 and Eigen are placed under the project in 'external' or 'libs' directories
add_subdirectory(libs/pybind11)


set(EIGEN_ROOT "$ENV{EIGEN_ROOT}")
if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT /usr/local/include/eigen3)
  message(STATUS "${EIGEN_ROOT}")
endif()
if (NOT EXISTS ${EIGEN_ROOT})
  message(FATAL_ERROR "Please point EIGEN_ROOT to your Eigen3 directory.")
endif()

# Include the headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python
    ${EIGEN_ROOT}
)

# Create the binding library
pybind11_add_module(BoxesMatch_cpp
    bindings/python/module.cpp
    BoxesMatch.cpp
    hungarian.cpp
    IoU_utils.cpp
)

# For the standalone executable
add_executable(BoxesMatchExecutable BoxesMatch.cpp hungarian.cpp IoU_utils.cpp)
# Link against any necessary libraries
target_link_libraries(BoxesMatchExecutable PRIVATE pybind11::embed)
# Add additional target_link_libraries() as needed
