cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_project)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")

find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/models)

add_executable(main main.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 17)
