// 声明通用初始解生成接口，供调度总入口调用。
#pragma once
#include "plan.hpp"

namespace npu {

// 生成和复核通用候选，选择搜索的初始解。
Solution generate_initial_plan(const Problem &graph, const Partition &partition, int cores);
}
