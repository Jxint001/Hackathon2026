#include "schedule.hpp"
#include "model.hpp"

#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace npu {

// 结合已测开始时间和原有依赖，生成合法参考顺序。
std::vector<int> reference_group_order(const SchedulingGroups& r, const Plan& plan, const Evaluation& score) {
    std::vector<std::vector<int>> next(r.units.size());
    for (int v : r.topo) next[v] = r.units[v].succ;
    for (const auto& order : plan.orders) {
        int last = -1;
        for (int op : order) {
            const int v = r.of[op];
            if (last >= 0 && last != v) next[last].push_back(v);
            last = v;
        }
    }
    std::vector<int> degree(r.units.size());
    for (auto& list : next) { unique_sort(list); for (int v : list) ++degree[v]; }
    using Item = std::pair<Cost, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> ready;
    for (int v : r.topo) if (!degree[v]) ready.emplace(score.op_start[r.units[v].ops[0]], v);
    std::vector<int> result;
    while (!ready.empty()) {
        const int v = ready.top().second; ready.pop(); result.push_back(v);
        for (int u : next[v]) if (!--degree[u]) ready.emplace(score.op_start[r.units[u].ops[0]], u);
    }
    return result.size() == r.units.size() ? result : r.dfs;
}

// 前驱都已排入顺序的组才可选，保持全局无环。
// 时间和内存状态仅作排序估计，完整方案再由模型评估。
std::vector<int> make_ready_order(const SchedulingGroups& r, const std::vector<int>& owner, int cores,
                         const std::vector<int>& reference, const ReadyPolicy& policy) {
    const int n = static_cast<int>(r.units.size());
    std::vector<int> rank(n), degree(n), cursor(cores);
    std::vector<std::vector<int>> base(cores);
    for (int v : reference) { rank[v] = static_cast<int>(base[owner[v]].size()); base[owner[v]].push_back(v); }
    std::vector<std::set<std::pair<int, int>>> ready(cores * kPipeCount);
    for (int v : r.topo) {
        degree[v] = static_cast<int>(r.units[v].pred.size());
        if (!degree[v]) ready[owner[v] * kPipeCount + r.units[v].pipe].emplace(rank[v], v);
    }
    std::vector<std::array<Cost, kPipeCount>> pipe_end(cores);
    std::vector<Cost> last_start(cores), finish(n);
    std::vector<char> done(n), resident(r.graph.tensors.size() * cores);
    std::vector<int> remaining(resident.size());
    std::vector<std::array<Cost, 2>> used(cores);
    for (int v : r.topo) for (int t : r.units[v].uses) ++remaining[t * cores + owner[v]];
    std::vector<std::vector<int>> future;
    std::vector<std::size_t> future_cursor;
    std::vector<std::set<std::pair<int, int>>> eviction;
    std::vector<int> next_rank;
    if (policy.bounded_residency) {
        future.resize(resident.size()); future_cursor.resize(resident.size());
        eviction.resize(cores * 2); next_rank.assign(resident.size(), INT32_MAX);
        for (int v : reference) for (int t : r.units[v].uses) future[t * cores + owner[v]].push_back(v);
    }
    std::vector<int> result; result.reserve(n);
    while (static_cast<int>(result.size()) < n) {
        int best = -1;
        Cost best_start = 0;
        std::tuple<double, double, int, int> best_key;
        for (const auto& queue : ready) {
            int inspected = 0;
            for (const auto& item : queue) {
                const int v = item.second, c = owner[v];
                // 查看有界就绪集合，允许复用优先策略扩大重排范围。
                if (inspected++ == policy.lookahead || (inspected > 1 && item.first >= cursor[c] + policy.window)) break;
                Cost start = last_start[c];
                for (int p = 0; p < kPipeCount; ++p) if (r.units[v].work[p]) start = std::max(start, pipe_end[c][p]);
                for (int p : r.units[v].pred) start = std::max(start, finish[p] + (owner[p] == c ? 0 : kCrossCoreDelay));
                std::array<Cost, 2> alloc{}, freed{};
                std::array<double, 2> progress{};
                for (int t : r.units[v].uses) {
                    const auto& tensor = r.graph.tensors[t];
                    if (tensor.storage < 0) continue;
                    const int storage = tensor.storage;
                    if (resident[t * cores + c]) progress[storage] += static_cast<double>(tensor.bytes) / std::max(1, remaining[t * cores + c]);
                    if (!resident[t * cores + c]) alloc[storage] += tensor.bytes;
                    if (remaining[t * cores + c] == 1) freed[storage] += tensor.bytes;
                }
                double pressure = 0, delta = 0, memory = 0;
                for (int p = 0; p < 2; ++p) {
                    pressure += policy.memory_weight[p] * std::max(0.0, static_cast<double>(used[c][p] + alloc[p]) / (policy.memory_fraction * kCapacity[p]) - 1);
                    delta += policy.memory_weight[p] * static_cast<double>(alloc[p] - freed[p]) / kCapacity[p];
                    memory += policy.memory_weight[p] * (policy.allocate * alloc[p] - policy.release * freed[p] - policy.progress * progress[p]) / kCapacity[p];
                }
                const auto key = std::make_tuple(policy.start_weight * start + policy.pipe_bias * (r.units[v].pipe != 2) + policy.pressure_weight * pressure + memory,
                                                 policy.rank_first ? static_cast<double>(rank[v] - cursor[c]) : delta, rank[v] - cursor[c], v);
                if (best < 0 || key < best_key) { best = v; best_start = start; best_key = key; }
            }
        }
        if (best < 0) return r.dfs;
        const int c = owner[best];
        ready[c * kPipeCount + r.units[best].pipe].erase({rank[best], best});
        finish[best] = best_start;
        for (int p = 0; p < kPipeCount; ++p) if (r.units[best].work[p]) {
            pipe_end[c][p] = best_start + r.units[best].work[p]; finish[best] = std::max(finish[best], pipe_end[c][p]);
        }
        last_start[c] = best_start;
        done[best] = 1;
        if (policy.bounded_residency) {
            std::array<Cost, 2> needed{};
            for (int t : r.units[best].uses) if (r.graph.tensors[t].storage >= 0 && !resident[t * cores + c])
                needed[r.graph.tensors[t].storage] += r.graph.tensors[t].bytes;
            // 下一次使用位置按参考顺序估计；换出后撤销驻留奖励。
            for (int p = 0; p < 2; ++p) {
                auto& victims = eviction[c * 2 + p];
                while (used[c][p] + needed[p] > policy.memory_fraction * kCapacity[p]) {
                    auto victim = victims.end();
                    for (auto it = victims.rbegin(); it != victims.rend(); ++it)
                        if (!std::binary_search(r.units[best].uses.begin(), r.units[best].uses.end(), it->second)) {
                            victim = std::prev(it.base()); break;
                        }
                    if (victim == victims.end()) break;
                    const int t = victim->second; victims.erase(victim);
                    resident[t * cores + c] = 0; used[c][p] -= r.graph.tensors[t].bytes;
                }
            }
        }
        for (int t : r.units[best].uses) {
            const auto& tensor = r.graph.tensors[t];
            if (tensor.storage < 0) continue;
            const int storage = tensor.storage, index = t * cores + c;
            if (policy.bounded_residency) eviction[c * 2 + storage].erase({next_rank[index], t});
            if (!resident[index]) { resident[index] = 1; used[c][storage] += tensor.bytes; }
            if (!--remaining[index]) { resident[index] = 0; used[c][storage] -= tensor.bytes; }
            else if (policy.bounded_residency) {
                auto& cursor = future_cursor[index]; const auto& uses = future[index];
                while (cursor < uses.size() && done[uses[cursor]]) ++cursor;
                next_rank[index] = cursor == uses.size() ? INT32_MAX : rank[uses[cursor]];
                eviction[c * 2 + storage].emplace(next_rank[index], t);
            }
        }
        result.push_back(best);
        while (cursor[c] < static_cast<int>(base[c].size()) && done[base[c][cursor[c]]]) ++cursor[c];
        for (int u : r.units[best].succ) if (!--degree[u]) ready[owner[u] * kPipeCount + r.units[u].pipe].emplace(rank[u], u);
    }
    return result;
}

// 将生产者移近同核消费者，缩短输出驻留。
// 仅跨越无依赖操作，比较输出收益与输入驻留代价。
Plan delay_grouped_producers(const SchedulingGroups& r, const Plan& base, double factor, bool expanding_only, bool single_successor_only,
                 int only_depth) {
    Plan result = base;
    const auto owner = r.owners(base);
    const int cores = static_cast<int>(base.orders.size());
    std::vector<std::vector<int>> orders(cores), users(r.graph.tensors.size());
    std::vector<char> seen(r.units.size());
    for (int c = 0; c < cores; ++c) for (int op : base.orders[c]) {
        const int v = r.of[op];
        if (orders[c].empty() || orders[c].back() != v) {
            // 单操作重排可能拆开组，此时保留原顺序。
            if (seen[v]) return base;
            seen[v] = 1; orders[c].push_back(v);
        }
    }
    for (int v : r.topo) for (int t : r.units[v].uses) users[t].push_back(v);
    std::vector<int> rank(r.units.size());
    for (int c = 0; c < cores; ++c) {
        auto& seq = orders[c];
        for (int i = 0; i < static_cast<int>(seq.size()); ++i) rank[seq[i]] = i;
        const auto visit = seq;
        for (auto it = visit.rbegin(); it != visit.rend(); ++it) {
            const int v = *it; const auto& unit = r.units[v];
            if (only_depth >= 0 && unit.depth != only_depth) continue;
            if (unit.succ.empty() || (single_successor_only && unit.succ.size() != 1)) continue;
            int target = static_cast<int>(seq.size());
            for (int u : unit.succ) { if (owner[u] != c) { target = -1; break; } target = std::min(target, rank[u]); }
            const int from = rank[v]; --target;
            if (target <= from) continue;
            Cost in_bytes = 0, out_bytes = 0;
            for (int t : unit.inputs) in_bytes += r.graph.tensors[t].bytes;
            for (int t : unit.outputs) out_bytes += r.graph.tensors[t].bytes;
            if (expanding_only && in_bytes >= out_bytes) continue;
            double cost = 0, benefit = static_cast<double>(out_bytes) * (target - from);
            for (int t : unit.inputs) {
                if (std::binary_search(unit.outputs.begin(), unit.outputs.end(), t)) continue;
                int last = -1;
                for (int u : users[t]) if (u != v && owner[u] == c) last = std::max(last, rank[u]);
                if (last >= 0) cost += static_cast<double>(r.graph.tensors[t].bytes) * std::max(0, target - std::max(from, last));
            }
            if (!expanding_only && benefit <= factor * cost) continue;
            std::rotate(seq.begin() + from, seq.begin() + from + 1, seq.begin() + target + 1);
            for (int i = from; i <= target; ++i) rank[seq[i]] = i;
        }
    }
    for (int c = 0; c < cores; ++c) {
        result.orders[c].clear();
        for (int v : orders[c]) result.orders[c].insert(result.orders[c].end(), r.units[v].ops.begin(), r.units[v].ops.end());
    }
    return result;
}


Plan delay_local_producers(const Graph& g, Plan plan, double factor) {
    std::vector<int> owner(g.ops.size(), -1), rank(g.ops.size());
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c)
        for (int i = 0; i < static_cast<int>(plan.orders[c].size()); ++i) {
            const int v = plan.orders[c][i]; owner[v] = c; rank[v] = i;
        }
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c) {
        auto& order = plan.orders[c];
        const auto visit = order;
        for (auto it = visit.rbegin(); it != visit.rend(); ++it) {
            const int v = *it, from = rank[v];
            if (g.succ[v].empty() || g.succ[v].size() > 3) continue;
            int target = static_cast<int>(order.size());
            for (int u : g.succ[v]) {
                if (owner[u] != c) { target = -1; break; }
                target = std::min(target, rank[u]);
            }
            if (--target <= from) continue;
            double benefit = 0, cost = 0;
            for (int t : g.ops[v].outputs) benefit += static_cast<double>(g.tensors[t].bytes) * (target - from);
            for (int t : g.ops[v].inputs) {
                if (std::binary_search(g.ops[v].outputs.begin(), g.ops[v].outputs.end(), t)) continue;
                int last = -1;
                for (int u : g.tensors[t].consumers) if (u != v && owner[u] == c) last = std::max(last, rank[u]);
                for (int u : g.tensors[t].producers) if (u != v && owner[u] == c) last = std::max(last, rank[u]);
                if (last >= 0) cost += static_cast<double>(g.tensors[t].bytes) * std::max(0, target - std::max(from, last));
            }
            if (benefit <= factor * cost) continue;
            std::rotate(order.begin() + from, order.begin() + from + 1, order.begin() + target + 1);
            for (int i = from; i <= target; ++i) rank[order[i]] = i;
        }
    }
    return plan;
}

bool compose_local_plan(const Graph& g, const GraphAnalysis& facts,
                        const Plan& base, const Plan& local, Plan& result) {
    if (base.orders.size() != local.orders.size()) return false;
    const int cores = static_cast<int>(base.orders.size());
    std::vector<int> owner(g.ops.size(), -1), rank(g.ops.size(), INT32_MAX), selected(g.ops.size());
    int covered = 0;
    for (int c = 0; c < cores; ++c) {
        for (int i = 0; i < static_cast<int>(base.orders[c].size()); ++i) {
            const int v = base.orders[c][i]; owner[v] = c; rank[v] = i;
        }
        for (int i = 0; i < static_cast<int>(local.orders[c].size()); ++i) {
            const int v = local.orders[c][i];
            if (v < 0 || v >= static_cast<int>(g.ops.size()) || g.ops[v].copy || selected[v]) return false;
            selected[v] = 1; owner[v] = c; rank[v] = i; ++covered;
        }
    }
    if (covered == static_cast<int>(g.work.size())) { result = local; return valid_plan(g, result); }
    // 外围的共写者跟随局部方案；局部方案自身的同核冲突直接拒绝。
    for (const auto& group : facts.co_groups) {
        int fixed = -1;
        for (int v : group) if (selected[v]) {
            if (fixed >= 0 && owner[v] != fixed) return false;
            fixed = owner[v];
        }
        if (fixed < 0) fixed = owner[group.front()];
        if (fixed < 0) return false;
        for (int v : group) owner[v] = fixed;
    }
    std::vector<std::vector<int>> next = g.succ;
    const auto add_sequence = [&](const std::vector<int>& order, bool keep_selected) {
        int previous = -1;
        for (int v : order) if (static_cast<bool>(selected[v]) == keep_selected) {
            if (previous >= 0) next[previous].push_back(v);
            previous = v;
        }
    };
    for (int c = 0; c < cores; ++c) {
        add_sequence(base.orders[c], false);
        add_sequence(local.orders[c], true);
    }
    std::vector<int> degree(g.ops.size());
    for (int v : g.work) { unique_sort(next[v]); for (int u : next[v]) ++degree[u]; }
    using Entry = std::pair<int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> ready;
    for (int v : g.work) if (!degree[v]) ready.emplace(rank[v], v);
    result.orders.assign(cores, {}); covered = 0;
    while (!ready.empty()) {
        const int v = ready.top().second; ready.pop();
        if (owner[v] < 0 || owner[v] >= cores) return false;
        result.orders[owner[v]].push_back(v); ++covered;
        for (int u : next[v]) if (!--degree[u]) ready.emplace(rank[u], u);
    }
    return covered == static_cast<int>(g.work.size()) && valid_plan(g, result);
}

} // npu 命名空间
