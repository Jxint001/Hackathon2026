#pragma once

#include "analysis.hpp"

namespace npu {

struct ReadyPolicy {
    int window = 512, lookahead = 64;
    double start_weight = 1, pipe_bias = 0, pressure_weight = 0;
    double allocate = 0, release = 0, progress = 0, memory_fraction = 0.85;
    bool rank_first = false;
    // 换出后撤销驻留标记；两种存储按换出压力赋权。
    bool bounded_residency = false;
    std::array<double, 2> memory_weight{{1, 1}};
};

struct Evaluation;
std::vector<int> reference_group_order(const SchedulingGroups& groups, const Plan& plan,
                                       const Evaluation& score);

std::vector<int> make_ready_order(const SchedulingGroups& groups, const std::vector<int>& owner, int cores,
                                  const std::vector<int>& reference, const ReadyPolicy& policy);
Plan delay_grouped_producers(const SchedulingGroups& groups, const Plan& plan, double factor,
                            bool expanding_only, bool single_successor_only, int only_depth = -1);
Plan delay_local_producers(const Graph& graph, Plan plan, double factor = 1.0);

// 保留局部特化的相对顺序，再按原图依赖接入外围操作。
bool compose_local_plan(const Graph& graph, const GraphAnalysis& facts,
                        const Plan& base, const Plan& local, Plan& result);

} // npu 命名空间
