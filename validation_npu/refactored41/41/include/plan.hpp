#pragma once
#include "graph.hpp"
#include <string>

namespace npu {
// 顺序投影到各核；不会强制不同核串行。
struct Plan {
    std::vector<int> core, order;
};
struct PlanScore {
    double makespan = 0, copy_bytes = 0;
};
struct SearchStatistics {
    int evaluations = 0;
};
struct Solution {
    Plan plan;
    PlanScore score;
    SearchStatistics statistics;
    Partition partition;
};
std::string serialize_plan(const Problem &problem, const Solution &solution, int cores);
}
