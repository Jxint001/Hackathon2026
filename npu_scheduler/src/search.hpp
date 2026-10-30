#pragma once

#include "model.hpp"

#include <map>
#include <functional>
#include <memory>

namespace npu {

struct Candidate {
    Plan plan;
    Evaluation score;
};

// 动作只构造方案，评估、保留和预算由统一搜索负责。
struct PlanAction {
    const char* name;
    std::function<Plan()> apply;
};
using PlanActions = std::vector<PlanAction>;

struct ActionProvider {
    std::size_t coverage = 0;
    std::function<PlanActions(const Plan&)> initial;
    std::function<PlanActions(const Candidate&, int)> neighbors;
};

struct GraphAnalysis;
struct SchedulingGroups;
Candidate search_actions(const Graph& graph, const GraphAnalysis& facts, const SchedulingGroups& groups, int cores, int budget,
                    const std::vector<ActionProvider>& providers);

bool better(const Candidate& a, const Candidate& b);
std::pair<std::uint64_t, std::uint64_t> fingerprint(const Plan& plan);

enum class StopRule { TimeBound, TimeAndCopyBounds };

// 全部策略共用去重、评估、最优记录和预算。
struct PlanSearchState {
    const Graph& graph;
    int cores, evaluated_count = 0, max_evaluations;
    Cost time_lower_bound, minimum_copy_bytes;
    StopRule stop_rule;
    Candidate best;
    using PlanKey = std::pair<std::uint64_t, std::uint64_t>;
    std::map<PlanKey, Cost> tried_bounds;
    std::map<PlanKey, Evaluation> cached_scores;

    PlanSearchState(const Graph& graph, int cores, int budget, StopRule rule);
    bool should_stop() const;
    Evaluation evaluate_candidate(const Plan& plan, const char* label = nullptr, Cost cutoff = -1);
};

} // npu 命名空间
