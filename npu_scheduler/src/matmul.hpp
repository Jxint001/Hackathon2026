#pragma once

#include "analysis.hpp"
#include "search.hpp"

namespace npu {

// 局部复用网格；每个格子记录按依赖相接的操作。
struct MatmulGrid {
    int rows = 0, columns = 0, depth = 0;
    std::vector<std::vector<int>> chains;
    Cost row_bytes = 0, column_bytes = 0, partial_bytes = 0;
    std::array<Cost, 2> row_pool{}, column_pool{}, partial_pool{};
    std::size_t op_count() const { return chains.size() * static_cast<std::size_t>(depth); }
};

std::vector<MatmulGrid> find_reuse_grids(const Graph& graph, const GraphAnalysis& analysis);
// 网格只提供块构造和变换，统一搜索负责选择。
ActionProvider grid_actions(const Graph& graph, const GraphAnalysis& analysis,
                            const MatmulGrid& grid, int cores);

} // npu 命名空间
