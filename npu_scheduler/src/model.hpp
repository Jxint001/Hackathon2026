#pragma once

#include "graph.hpp"

namespace npu {

struct Evaluation {
    bool valid = false;
    Cost makespan = kInfinity;
    Cost ddr_cycles = 0;
    Cost bytes = 0;
    Cost spill_bytes = 0;
    // 搜索诊断数据，不写入提交方案。
    std::vector<Cost> op_start, op_end, core_end, tensor_spill_bytes;
    std::string error;
};

// 按实际生产核、消费核组合统计张量搬运。
Cost partition_lower_bound(const Graph& graph, const Plan& plan);
Cost partition_ddr_cycles(const Graph& graph, const Plan& plan);
Cost universal_lower_bound(const Graph& graph, int cores);
Evaluation evaluate(const Graph& graph, const Plan& plan,
                    Cost incumbent = kInfinity);

} // npu 命名空间
