#pragma once

#include "graph.hpp"

namespace npu {
// Conv 等非规则图的区域分核与排序搜索。
Plan solve_general(const Graph& graph, int cores);
}
