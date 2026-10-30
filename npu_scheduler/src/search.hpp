#pragma once

#include "model.hpp"

#include <set>

namespace npu {

struct Candidate {
    Plan plan;
    Evaluation score;
};

bool better(const Candidate& a, const Candidate& b);
std::pair<std::uint64_t, std::uint64_t> fingerprint(const Plan& plan);

enum class StopRule { TimeBound, TimeAndCopyBounds };

// 两个分支共用去重、评估和最优记录，各自指定预算与停止规则。
struct PlanSearchState {
    const Graph& graph;
    int cores, evaluated_count = 0, max_evaluations;
    Cost time_lower_bound, minimum_copy_bytes;
    StopRule stop_rule;
    Candidate best;
    std::set<std::pair<std::uint64_t, std::uint64_t>> tried_plan_hashes;

    PlanSearchState(const Graph& graph, int cores, int budget, StopRule rule);
    bool should_stop() const;
    Evaluation evaluate_candidate(const Plan& plan, const char* label = nullptr);
};

} // npu 命名空间
