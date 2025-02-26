# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(NLA_Project LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Release
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Cmake accelerate
set(CMAKE_CXX_FLAGS_RELEASE -Ofast)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")

# Find the Armadillo package
find_package(Armadillo REQUIRED)

# Add the include directory
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(include)

# Add source files
set(SOURCES
    src/main.cpp
)

# Define the executable target
add_executable(main ${SOURCES})

# Link Armadillo library
target_link_libraries(main PRIVATE ${ARMADILLO_LIBRARIES})