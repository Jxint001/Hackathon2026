# 文件核心功能：使用同一套 CMake 配置构建 Linux/Windows 的提交程序。
# 输入：包内 src 实现与 include 头文件及 CMake 构建选项；输出：build/bin/npu_sched（Windows 带 .exe）。
cmake_minimum_required(VERSION 3.16)
project(npu_sched LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 输出目录相对于当前构建目录，避免任何绝对路径配置。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} bin)
endforeach()
# 仅链接系统线程运行库，不使用提交要求禁止的包查找或下载机制。
add_library(npu_thread_support INTERFACE)
if(UNIX)
  target_compile_options(npu_thread_support INTERFACE -pthread)
  target_link_options(npu_thread_support INTERFACE -pthread)
endif()

# 主程序复用核心库；包内不声明缺失的工具和测试目标。
add_library(npu_core STATIC
  src/graph.cpp
  src/plan.cpp
  src/initial_plan.cpp
  src/plan_search.cpp
  src/search/structure_analysis.cpp
  src/search/candidate_generator.cpp
  src/search/candidate_evaluator.cpp
  src/search/search_controller.cpp
  src/search/grid_search.cpp
  src/search/spatial_search.cpp
  src/search/order_search.cpp
  src/search/multistart_search.cpp
  src/search/chain_search.cpp
  src/plan_evaluator.cpp
  src/evaluation/task_builder.cpp
  src/evaluation/local_schedule.cpp
  src/evaluation/memory_schedule.cpp
  src/evaluation/global_simulation.cpp
  src/scheduler.cpp
)
target_include_directories(npu_core PUBLIC include)
target_link_libraries(npu_core PUBLIC npu_thread_support)
if(MSVC)
  target_compile_options(npu_core PUBLIC /W4 /utf-8 /permissive-)
else()
  target_compile_options(npu_core PUBLIC -Wall -Wextra -Wpedantic)
endif()
add_executable(npu_sched src/main.cpp)
target_link_libraries(npu_sched PRIVATE npu_core)

