#pragma once
#include "search/candidate.hpp"
#include <map>
#include <optional>
#include <set>
namespace npu {
namespace search_detail {
// 每个起点独享状态；只显式复制已完成的有效评估缓存。
struct SearchState {
    Solution best;
    std::set<std::uint64_t> seen_candidates;
    std::map<std::uint64_t, PlanEvaluation> evaluation_cache, prefetched_scores;
    std::optional<ExecutionTrace> best_trace;
    std::size_t best_revision = 0;
    int phase_attempts = 0, total_attempts = 0;
    const int phase_limit;
    std::vector<SearchSeed> seeds;
    bool collect_seeds = false;
    SearchState(const Solution &baseline, int limit) : best(baseline), phase_limit(limit) {}
};
}
}
