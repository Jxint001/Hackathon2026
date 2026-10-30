#include "search.hpp"

#include <tuple>
#ifdef NPU_SEARCH_TRACE
#include <iostream>
#endif

namespace npu {
namespace {

Cost mandatory_bytes(const Graph& g) {
    Cost result = 0;
    for (const auto& t : g.tensors) {
        bool producer = false, consumer = false, output = false;
        for (int op : t.producers) producer = producer || !g.ops[op].copy;
        for (int op : t.consumers) {
            consumer = consumer || !g.ops[op].copy;
            output = output || g.ops[op].copy == 2;
        }
        if ((!producer && consumer) || (producer && (output || !consumer))) {
            if (result > kInfinity - t.bytes) return kInfinity;
            result += t.bytes;
        }
    }
    return result;
}

} // 匿名命名空间

bool better(const Candidate& a, const Candidate& b) {
    return std::tie(a.score.makespan, a.score.bytes, a.plan.orders) <
           std::tie(b.score.makespan, b.score.bytes, b.plan.orders);
}

std::pair<std::uint64_t, std::uint64_t> fingerprint(const Plan& plan) {
    std::uint64_t a = 1469598103934665603ULL, b = 0x9e3779b97f4a7c15ULL;
    for (const auto& order : plan.orders) {
        a = (a ^ UINT64_MAX) * 1099511628211ULL;
        b = b * 6364136223846793005ULL + order.size();
        for (int op : order) {
            a = (a ^ static_cast<std::uint64_t>(op)) * 1099511628211ULL;
            b = (b ^ a) * 2862933555777941757ULL + 3037000493ULL;
        }
    }
    return {a, b};
}

PlanSearchState::PlanSearchState(const Graph& g, int n, int budget, StopRule rule)
    : graph(g), cores(n), max_evaluations(budget),
      time_lower_bound(universal_lower_bound(g, n)),
      minimum_copy_bytes(rule == StopRule::TimeAndCopyBounds ? mandatory_bytes(g) : 0),
      stop_rule(rule) {}

bool PlanSearchState::should_stop() const {
    return evaluated_count >= max_evaluations ||
           (best.score.valid && best.score.makespan == time_lower_bound &&
            (stop_rule == StopRule::TimeBound || best.score.bytes == minimum_copy_bytes));
}

// 去重和下界筛选后评估；无效候选不进入后续搜索。
Evaluation PlanSearchState::evaluate_candidate(const Plan& plan, const char* label) {
    Evaluation result;
    if (should_stop() || !tried_plan_hashes.insert(fingerprint(plan)).second) return result;
    if (partition_lower_bound(graph, plan) > best.score.makespan) return result;
    ++evaluated_count;
    result = evaluate(graph, plan, best.score.makespan);
#ifdef NPU_SEARCH_TRACE
    if (label) std::cerr << evaluated_count << ' ' << label << " time=" << result.makespan
                         << " bytes=" << result.bytes << " error=" << result.error << '\n';
#else
    (void)label;
#endif
    if (result.valid && (!best.score.valid ||
        std::tie(result.makespan, result.bytes, plan.orders) <
        std::tie(best.score.makespan, best.score.bytes, best.plan.orders))) {
        best = Candidate{plan, result};
        // 已接近时间下界时，缩减后续评估次数。
        if (static_cast<double>(best.score.makespan - time_lower_bound) <= 0.005 * best.score.makespan)
            max_evaluations = std::min(max_evaluations, std::max(24, evaluated_count + 8));
    }
    return result;
}

} // npu 命名空间
