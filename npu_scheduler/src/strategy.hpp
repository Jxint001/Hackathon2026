#pragma once

#include "graph.hpp"

namespace npu {
Plan solve(const Graph& graph, int cores);
} // namespace npu
