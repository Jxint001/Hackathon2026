#include "search.hpp"
#include "blocks.hpp"
#include "schedule.hpp"

#include <set>
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
Evaluation PlanSearchState::evaluate_candidate(const Plan& plan, const char* label, Cost cutoff) {
    Evaluation result;
    if (should_stop()) return result;
    const auto key = fingerprint(plan);
    const bool local_search = cutoff >= 0;
    if (!local_search) cutoff = best.score.makespan;
    const auto cached = cached_scores.find(key);
    if (cached != cached_scores.end())
        return local_search && cached->second.makespan <= cutoff ? cached->second : result;
    const auto tried = tried_bounds.find(key);
    if (tried != tried_bounds.end() && tried->second >= cutoff) return result;
    // 不同起点可使用各自的剪枝界，避免较差种子尚未改进就被全局最优淘汰。
    tried_bounds[key] = cutoff;
    if (partition_lower_bound(graph, plan) > cutoff) return result;
    ++evaluated_count;
    result = evaluate(graph, plan, cutoff);
    if (result.valid) cached_scores.emplace(key, result);
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

namespace {

struct PoolEntry {
    Candidate candidate;
    int source;
    std::vector<int> owner;
};

std::vector<int> plan_owners(const Graph& graph, const Plan& plan) {
    std::vector<int> owner(graph.ops.size(), -1);
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c)
        for (int v : plan.orders[c]) owner[v] = c;
    return owner;
}

// 保留来源和分核的差异，避免一个起点过早淘汰另一个起点。
void retain(std::vector<PoolEntry>& pool, PoolEntry entry) {
    const auto key = fingerprint(entry.candidate.plan);
    for (const auto& old : pool) if (old.source == entry.source && fingerprint(old.candidate.plan) == key) return;
    pool.push_back(std::move(entry));
    std::stable_sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) { return better(a.candidate, b.candidate); });
    std::vector<PoolEntry> kept;
    std::set<int> sources;
    std::map<std::pair<int, std::vector<int>>, int> owners;
    for (auto& item : pool) {
        const bool first = sources.insert(item.source).second;
        auto& count = owners[{item.source, item.owner}];
        if (first || (kept.size() < 8 && count == 0)) { ++count; kept.push_back(std::move(item)); }
    }
    pool = std::move(kept);
}

PlanActions shared_block_actions(const Graph& graph, const GraphAnalysis& facts, const SchedulingGroups& groups,
                                 const Candidate& seed, int round) {
    PlanActions actions;
    const auto base = std::make_shared<Plan>(seed.plan);
    const int cores = static_cast<int>(base->orders.size());
    const auto owner = groups.owners(*base);
    const auto reference = reference_group_order(groups, *base, seed.score);
    std::array<double, 2> pressure{};
    for (int t = 0; t < static_cast<int>(graph.tensors.size()); ++t)
        if (graph.tensors[t].storage >= 0) pressure[graph.tensors[t].storage] += seed.score.tensor_spill_bytes[t];
    for (bool bounded : {true, false}) actions.push_back({"reuse-ready", [&groups, owner, reference, pressure, bounded, cores] {
        ReadyPolicy policy; policy.window = 4096; policy.lookahead = 64;
        policy.start_weight = 1.0 / kCrossCoreDelay; policy.pressure_weight = 1;
        policy.allocate = policy.release = 1; policy.progress = 4; policy.memory_fraction = .9;
        policy.rank_first = true; policy.bounded_residency = bounded;
        const double maximum = std::max(1.0, std::max(pressure[0], pressure[1]));
        for (int p = 0; p < 2; ++p) policy.memory_weight[p] = std::max(.01, pressure[p] / maximum);
        auto plan = groups.project(owner, make_ready_order(groups, owner, cores, reference, policy), cores);
        return delay_grouped_producers(groups, plan, 1, true, false);
    }});
    auto targets = spill_targets(graph, seed.plan, seed.score, 6);
    // 同一动作作用于所有结构，拆分大小与推进深度由参数决定。
    for (int t : targets) {
        TensorBatch config; config.tensors = {t}; config.batch = round % 2 ? 16 : 4;
        config.cone_limit = 512; config.steps = round % 2; config.width = round % 2 + 1;
        config.late = round % 2;
        const auto blocks = std::make_shared<CoreBlocks>();
        actions.push_back({"pack-consumers", [&graph, &facts, base, config, blocks] {
            if (blocks->empty()) *blocks = consumer_blocks(graph, facts, *base, config);
            return place_blocks(graph, *base, *blocks, config.width, config.steps <= 0 ? INT32_MAX : config.steps, config.late);
        }});
        const int light = static_cast<int>(std::min_element(seed.score.core_end.begin(), seed.score.core_end.end()) - seed.score.core_end.begin());
        // 只提议少量块迁移，完整时间模型决定是否接受。
        actions.push_back({"move-consumer-block", [&graph, &facts, base, config, blocks, light, cores] {
            if (blocks->empty()) *blocks = consumer_blocks(graph, facts, *base, config);
            for (int c = 0; c < cores; ++c) if (c != light && !(*blocks)[c].empty())
                return move_block(graph, facts, *base, (*blocks)[c].back(), light);
            return Plan{};
        }});
    }
    actions.push_back({"delay-operations", [&graph, base] { return delay_local_producers(graph, *base); }});
    // 初轮检查整体重排；后续优先用实测换出量调整局部块。
    if (round > 0) std::rotate(actions.begin(), actions.begin() + 2, actions.end());
    return actions;
}

} // 匿名命名空间

Candidate search_actions(const Graph& graph, const GraphAnalysis& facts, const SchedulingGroups& groups, int cores, int budget,
                    const std::vector<ActionProvider>& providers) {
    PlanSearchState state(graph, cores, budget, StopRule::TimeAndCopyBounds);
    for (bool dfs : {true, false}) {
        Plan plan; plan.orders.resize(cores); plan.orders[0] = dependency_order(graph, dfs);
        state.evaluate_candidate(plan, "baseline");
    }
    std::vector<PoolEntry> pool;
    std::vector<Cost> best_by_source(providers.size(), kInfinity);
    const auto consider = [&](const PlanAction& action, int source, Cost cutoff) {
        if (state.should_stop()) return;
        auto plan = action.apply();
        if (plan.orders.size() != static_cast<std::size_t>(cores)) return;
        auto score = state.evaluate_candidate(plan, action.name, cutoff);
        if (!score.valid) return;
        best_by_source[source] = std::min(best_by_source[source], score.makespan);
        auto owner = plan_owners(graph, plan);
        retain(pool, PoolEntry{Candidate{std::move(plan), std::move(score)}, source, std::move(owner)});
    };
    std::size_t total_coverage = 0;
    for (const auto& provider : providers) total_coverage += provider.coverage;
    const int seed_budget = budget * 3 / 5;
    // 所有结构只提交动作，统一入口完成预算、评估与候选保留。
    for (int source = 0; source < static_cast<int>(providers.size()) && !state.should_stop(); ++source) {
        const auto& provider = providers[source];
        if (!provider.coverage) continue;
        const int allowance = std::max(12, static_cast<int>(seed_budget * provider.coverage / std::max<std::size_t>(1, total_coverage)));
        const int end = std::min(state.max_evaluations, state.evaluated_count + allowance);
        const auto actions = provider.initial(state.best.plan);
        for (const auto& action : actions) {
            if (state.should_stop() || state.evaluated_count >= end) break;
            const Cost local = best_by_source[source];
            consider(action, source, local == kInfinity ? local : local + local / 20);
        }
    }
    // 同一轮允许先改块内顺序，下一轮再迁块或重分块。
    std::set<std::tuple<std::uint64_t, std::uint64_t, int, int>> expanded;
    for (int round = 0; round < 6 && !state.should_stop() && !pool.empty(); ++round) {
        auto seeds = pool;
        std::vector<PoolEntry> selected;
        std::map<int, int> sources;
        for (const auto& seed : seeds) if (sources[seed.source]++ < 2) {
            selected.push_back(seed);
            if (selected.size() == 4) break;
        }
        const auto previous = fingerprint(state.best.plan);
        for (const auto& seed : selected) {
            if (state.should_stop()) break;
            const auto key = fingerprint(seed.candidate.plan);
            if (!expanded.emplace(key.first, key.second, seed.source, std::min(round, 2)).second) continue;
            auto specialized = providers[seed.source].neighbors(seed.candidate, round);
            auto shared = shared_block_actions(graph, facts, groups, seed.candidate, round);
            std::vector<PlanActions> other;
            for (int source = 0; source < static_cast<int>(providers.size()); ++source)
                if (source != seed.source) other.push_back(providers[source].neighbors(seed.candidate, round));
            std::vector<std::size_t> position(other.size());
            const int remaining = state.max_evaluations - state.evaluated_count;
            const int allowance = std::max(12, remaining / std::max(2, static_cast<int>(selected.size()) * (round < 2 ? 2 : 1)));
            const int end = state.evaluated_count + allowance;
            std::size_t specific = 0, common = 0, turn = 0;
            while (!state.should_stop() && state.evaluated_count < end) {
                bool pending = specific < specialized.size() || common < shared.size();
                for (std::size_t i = 0; i < other.size(); ++i) pending = pending || position[i] < other[i].size();
                if (!pending) break;
                // 来源只影响优先级；其他结构和公共动作都能继续修改候选。
                for (int i = 0; i < 4 && specific < specialized.size() && state.evaluated_count < end; ++i)
                    consider(specialized[specific++], seed.source, seed.candidate.score.makespan + seed.candidate.score.makespan / 20);
                if (common < shared.size() && state.evaluated_count < end)
                    consider(shared[common++], seed.source, seed.candidate.score.makespan + seed.candidate.score.makespan / 20);
                if (!other.empty() && state.evaluated_count < end) {
                    const std::size_t i = turn++ % other.size();
                    if (position[i] < other[i].size())
                        consider(other[i][position[i]++], seed.source, seed.candidate.score.makespan + seed.candidate.score.makespan / 20);
                }
            }
        }
        if (round >= 2 && fingerprint(state.best.plan) == previous) break;
    }
    return std::move(state.best);
}

} // npu 命名空间
