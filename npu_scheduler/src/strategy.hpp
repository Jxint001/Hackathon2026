#pragma once

#include "graph.hpp"

namespace npu {
// 下标 q 保存最多使用 q 核的方案，下标 0、1 留空。
std::vector<Plan> solve_core_limits(const Graph& graph, int cores);
Plan solve(const Graph& graph, int cores);
} // npu 命名空间
