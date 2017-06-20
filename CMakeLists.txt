cmake_minimum_required(VERSION 3.7)
project(LinearRegressionProject)


set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


include_directories(include/)
include_directories(src/)
include_directories("include/eigen")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*h")

add_executable(LinearRegressionProject ${SOURCES} ${HEADERS})
target_link_libraries (LinearRegressionProject Eigen3::Eigen)
