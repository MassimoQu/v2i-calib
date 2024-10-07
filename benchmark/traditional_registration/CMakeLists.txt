cmake_minimum_required(VERSION 3.16)
project(registration)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(registration SHARED registration.cpp)
target_link_libraries(registration ${PCL_LIBRARIES})
