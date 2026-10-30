cmake_minimum_required(VERSION 3.16)
project(npu_scheduler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(npu_sched
    src/main.cpp
    src/graph.cpp
    src/model.cpp
    src/analysis.cpp
    src/blocks.cpp
    src/schedule.cpp
    src/search.cpp
    src/general.cpp
    src/matmul.cpp
    src/strategy.cpp)
target_include_directories(npu_sched PRIVATE third_party)
set_target_properties(npu_sched PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" upper_config)
    set_target_properties(npu_sched PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/bin")
endforeach()
if(MSVC)
    target_compile_options(npu_sched PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_compile_definitions(npu_sched PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(npu_sched PRIVATE -Wall -Wextra -Wpedantic -pthread)
    target_link_options(npu_sched PRIVATE -pthread)
endif()
