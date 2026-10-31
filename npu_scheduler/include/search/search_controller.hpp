#pragma once
#include "search/candidate_evaluator.hpp"
#include "search/candidate_generator.hpp"
#include "search/search_state.hpp"
namespace npu {
namespace search_detail {
// 唯一负责预算、缓存、起点选择和接受候选的组件。
class SearchController {
    const Problem &problem;
    int cores, size;
    const CandidateGenerator generator;
    const CandidateEvaluator evaluator;
    SearchState state;
    bool better(const PlanEvaluation &r) const;
    void remember_seed(const Candidate &candidate, const PlanEvaluation &score);
    void prefetch(const std::vector<Candidate> &candidates);
    PlanEvaluation consider(const Sequences &seq, int chunk = 1);
    void consider_batch(const std::vector<Candidate> &candidates);
    const ExecutionTrace &trace_best();
    bool competitive(const Sequences &seq) const;
    void grid_search(const Grid &grid);
    void granularity();
    void producer_search();
    void overlap_search();
    void convolution_orders();
    void copy_completed_to(SearchController &target) const;
    void alternate_seeds();
    void multistart_search(std::uint64_t starting_hash);
    void aligned_tiles();
    void matrix_starts();
    void chain_moves();
    void capacity_tiles();
    void grid_candidates();
    void phase(SearchStage step);

  public:
    Solution run();
    Solution finish();
    SearchController(const Problem &g, int n, const Solution &baseline);
};
}
}
