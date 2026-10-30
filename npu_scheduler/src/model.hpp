#pragma once

#include "graph.hpp"

namespace npu {

struct Evaluation {
    bool valid = false;
    Cost makespan = kInfinity;
    Cost ddr_cycles = 0;
    Cost bytes = 0;
    Cost spill_bytes = 0;
    // Search diagnostics, never included in the submitted plan.
    std::vector<Cost> op_start, op_end, core_end, tensor_spill_bytes;
    std::string error;
};

// Tensor communication is counted per actual producer/consumer core pair.
Cost partition_lower_bound(const Graph& graph, const Plan& plan);
Cost partition_ddr_cycles(const Graph& graph, const Plan& plan);
Cost universal_lower_bound(const Graph& graph, int cores);
Evaluation evaluate(const Graph& graph, const Plan& plan,
                    Cost incumbent = kInfinity);

} // namespace npu
