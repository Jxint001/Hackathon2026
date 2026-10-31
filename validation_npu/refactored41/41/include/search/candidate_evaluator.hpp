#pragma once
#include "search/candidate.hpp"
namespace npu {
namespace search_detail {
// 无共享可写状态，允许在线程池中并行复核候选。
class CandidateEvaluator {
    const Problem &problem;
    int cores, size;

  public:
    CandidateEvaluator(const Problem &p, int n)
        : problem(p), cores(n), size(static_cast<int>(p.ops.size())) {}
    Solution materialize_solution(const Sequences &seq, int chunk) const;
    std::uint64_t candidate_hash(const Sequences &seq, int chunk) const;
    PlanEvaluation evaluate(const Candidate &candidate) const;
};
}
}
