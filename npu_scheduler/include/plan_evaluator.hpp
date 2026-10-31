// 定义候选复核成绩和执行轨迹，声明仅供内部选优使用的评估接口。
#pragma once
#include "plan.hpp"

namespace npu {

// 内部执行复核返回的候选成绩，不是官方最终评分。
struct PlanEvaluation {
    bool valid = false;
    double makespan = 0;
    double copy_bytes = 0;
    double spill_bytes = 0;
};
// 可选的候选执行轨迹，供重排、迁核和多起点搜索分析使用。
struct ExecutionTrace {

    std::vector<std::vector<int>> sequences;
    std::vector<double> starts;
    std::vector<double> core_ends;
};
// 只读取操作分组映射和分核顺序，不要求为待筛选候选构造完整 Partition。
PlanEvaluation evaluate_plan(const Problem &graph, const std::vector<int> &mapping, const Plan &solution,
                             int cores, ExecutionTrace *trace = nullptr);

}
