#pragma once

#include "analysis.hpp"
#include "model.hpp"

namespace npu {

// 块可拆分、交错；每层是本轮一起推进的操作。
struct ComputeBlock {
    std::vector<std::vector<int>> stages;
};
using CoreBlocks = std::vector<std::vector<ComputeBlock>>;

Plan emit_blocks(const CoreBlocks& blocks, int width, int steps);

// 把块顺序作为优先级，按原图依赖完成整个方案。
Plan place_blocks(const Graph& graph, const Plan& base, const CoreBlocks& blocks,
                  int width, int steps, bool late = false);

struct TensorBatch {
    std::vector<int> tensors;
    int batch = 8, cone_limit = 512, width = 1, steps = 1;
    bool late = false;
};

std::vector<int> spill_targets(const Graph& graph, const Plan& plan,
                               const Evaluation& score, int limit);
CoreBlocks consumer_blocks(const Graph& graph, const GraphAnalysis& facts,
                           const Plan& base, const TensorBatch& config);
Plan pack_consumers(const Graph& graph, const GraphAnalysis& facts,
                    const Plan& base, const TensorBatch& config);

// 迁移时补齐同核约束，块内顺序仍可单独调整。
Plan move_block(const Graph& graph, const GraphAnalysis& facts, const Plan& base,
                const ComputeBlock& block, int destination);

} // npu 命名空间
