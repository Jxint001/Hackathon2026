#include "search/candidate_evaluator.hpp"
#include <algorithm>
namespace npu {
namespace search_detail {
Solution CandidateEvaluator::materialize_solution(const Sequences &seq, int chunk) const {
    Solution solution;
    std::vector<std::vector<int>> group_ops;
    for (int core = 0; core < static_cast<int>(seq.size()); ++core) {
        const auto &operations = seq[core];
        for (int begin = 0; begin < static_cast<int>(operations.size()); begin += chunk) {
            int end = std::min(begin + chunk, static_cast<int>(operations.size()));
            int group = static_cast<int>(group_ops.size());
            group_ops.emplace_back(operations.begin() + begin, operations.begin() + end);
            solution.plan.core.push_back(core);
            solution.plan.order.push_back(group);
        }
    }
    solution.partition = make_partition(problem, std::move(group_ops));
    return solution;
}
std::uint64_t CandidateEvaluator::candidate_hash(const Sequences &seq, int chunk) const {

    std::uint64_t hash = 1469598103934665603ULL;
    // 把一个整数混入当前候选哈希。
    auto mix = [&](std::uint64_t x) {
        hash ^= x;
        hash *= 1099511628211ULL;
    };
    mix(static_cast<unsigned>(chunk));
    for (const auto &part : seq) {
        mix(static_cast<std::uint64_t>(size) + 1);
        for (int u : part) {
            mix(static_cast<unsigned>(u));
        }
    }
    return hash;
}
PlanEvaluation CandidateEvaluator::evaluate(const Candidate &candidate) const {

    std::vector<int> mapping(size, -1);
    Plan plan;
    for (int c = 0; c < cores; ++c) {
        const auto &part = candidate.seq[c];
        for (int i = 0; i < static_cast<int>(part.size()); i += candidate.chunk) {
            int group = static_cast<int>(plan.core.size());
            plan.core.push_back(c);
            plan.order.push_back(group);
            for (int j = i; j < std::min(i + candidate.chunk, static_cast<int>(part.size())); ++j) {
                mapping[part[j]] = group;
            }
        }
    }
    return evaluate_plan(problem, mapping, plan, cores);
}
}
}
