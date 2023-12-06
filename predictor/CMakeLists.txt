cmake_minimum_required(VERSION 3.16)
project(predictor)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanobind)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/tensorflow/lite
    ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite EXCLUDE_FROM_ALL
)

nanobind_add_module(
    predictor
    src/predictor.cpp
    src/power_model.cpp
    src/load_model.cpp
)
target_link_libraries(predictor PRIVATE tensorflow-lite)
