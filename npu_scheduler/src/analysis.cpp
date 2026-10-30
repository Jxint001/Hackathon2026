#include "analysis.hpp"

#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace npu {

GraphAnalysis::GraphAnalysis(const Graph& g)
    : co_group(g.ops.size(), -1), depth(g.ops.size()), reader_count(g.tensors.size()), writer_count(g.tensors.size()) {
    UnionFind writers(static_cast<int>(g.ops.size()));
    for (int t = 0; t < static_cast<int>(g.tensors.size()); ++t) {
        int first = -1;
        for (int v : g.tensors[t].producers) if (!g.ops[v].copy) {
            ++writer_count[t];
            if (first < 0) first = v; else writers.join(first, v);
        }
        for (int v : g.tensors[t].consumers) if (!g.ops[v].copy) ++reader_count[t];
    }
    std::map<int, int> groups;
    for (int v : g.work) {
        auto inserted = groups.emplace(writers.find(v), static_cast<int>(co_groups.size()));
        if (inserted.second) co_groups.emplace_back();
        co_group[v] = inserted.first->second;
        co_groups[co_group[v]].push_back(v);
    }
    for (int v : g.topo) for (int p : g.pred[v]) depth[v] = std::max(depth[v], depth[p] + 1);
}

double median_position(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return (v[(v.size() - 1) / 2] + v[v.size() / 2]) * 0.5;
}

// 含 COPY 求默认遍历序，再保留可分配的计算节点。
std::vector<int> reference_order_with_copies(const Graph& g) {
    std::vector<std::vector<int>> pred(g.ops.size()), succ(g.ops.size());
    for (const auto& t : g.tensors) for (int p : t.producers) for (int u : t.consumers) pred[u].push_back(p);
    for (const auto& edge : g.direct) pred[edge.second].push_back(edge.first);
    for (int v = 0; v < static_cast<int>(g.ops.size()); ++v) {
        unique_sort(pred[v]); for (int p : pred[v]) succ[p].push_back(v);
    }
    std::vector<int> degree(g.ops.size()), depth(g.ops.size()), queue;
    for (int v = 0; v < static_cast<int>(g.ops.size()); ++v) { degree[v] = static_cast<int>(pred[v].size()); if (!degree[v]) queue.push_back(v); }
    for (std::size_t i = 0; i < queue.size(); ++i) for (int u : succ[queue[i]]) {
        depth[u] = std::max(depth[u], depth[queue[i]] + 1); if (!--degree[u]) queue.push_back(u);
    }
    const auto key = [&](int v, int kind) { return std::make_tuple(g.ops[v].copy != kind, depth[v], -static_cast<std::int64_t>(v)); };
    std::vector<int> stack, result;
    std::vector<char> visited(g.ops.size());
    for (int v : queue) if (succ[v].empty()) stack.push_back(v);
    std::sort(stack.begin(), stack.end(), [&](int a, int b) { return key(a, 2) < key(b, 2); });
    while (!stack.empty()) {
        const int v = stack.back();
        if (visited[v]) { stack.pop_back(); continue; }
        std::vector<int> pending;
        for (int p : pred[v]) if (!visited[p]) pending.push_back(p);
        if (pending.empty()) {
            visited[v] = 1; stack.pop_back(); if (!g.ops[v].copy) result.push_back(v);
        } else {
            std::sort(pending.begin(), pending.end(), [&](int a, int b) { return key(a, 1) < key(b, 1); });
            stack.insert(stack.end(), pending.begin(), pending.end());
        }
    }
    return result;
}

// 用输入重叠推断相对位置编号，供按位置切段分核使用。
std::vector<double> infer_locality_axis(const Graph& g) {
    std::vector<int> level(g.ops.size());
    std::vector<int> first;
    // 取各依赖路径上首批 cycles > 1 的 PIPE_M 操作。
    for (int v : g.topo) {
        for (int p : g.pred[v]) level[v] = std::max(level[v], level[p]);
        if (g.ops[v].pipe == 2 && g.ops[v].cycles > 1) {
            if (!level[v]) first.push_back(v);
            ++level[v];
        }
    }
    std::vector<int> frequency(g.tensors.size());
    for (int v : first) for (int t : g.ops[v].inputs) ++frequency[t];
    // 按周期数和高频共享输入分组，避免混合不同计算族。
    std::map<std::pair<Cost, std::vector<int>>, std::vector<int>> families;
    for (int v : first) {
        std::vector<int> common;
        for (int t : g.ops[v].inputs)
            if (frequency[t] >= std::max(4, static_cast<int>(first.size()) / 20)) common.push_back(t);
        families[{g.ops[v].cycles, common}].push_back(v);
    }
    std::vector<int> best_path;
    for (const auto& family : families) {
        const auto& ops = family.second;
        if (ops.size() < 4) continue;
        std::map<int, int> count;
        for (int v : ops) for (int t : g.ops[v].inputs) ++count[t];
        // 保留局部复用输入，排除仅用一次或大范围共享的输入。
        std::set<int> local;
        for (const auto& item : count)
            if (item.second > 1 && item.second * 2 < static_cast<int>(ops.size())) local.insert(item.first);
        if (local.size() <= best_path.size()) continue;
        std::map<std::pair<int, int>, int> pairs;
        int strongest = 0;
        for (int v : ops) {
            std::vector<int> inputs;
            for (int t : g.ops[v].inputs) if (local.count(t)) inputs.push_back(t);
            // 过宽的输入集合不用于推断滑动窗口。
            if (inputs.size() > 32) continue;
            for (std::size_t a = 0; a < inputs.size(); ++a)
                for (std::size_t b = a + 1; b < inputs.size(); ++b)
                    strongest = std::max(strongest, ++pairs[{inputs[a], inputs[b]}]);
        }
        // 最常共同使用的输入视为邻居，要求连成一条完整路径。
        std::map<int, std::vector<int>> adjacent;
        for (const auto& edge : pairs) if (edge.second == strongest) {
            adjacent[edge.first.first].push_back(edge.first.second);
            adjacent[edge.first.second].push_back(edge.first.first);
        }
        std::vector<int> endpoints;
        bool path_graph = adjacent.size() == local.size();
        for (const auto& item : adjacent) {
            if (item.second.size() == 1) endpoints.push_back(item.first);
            else if (item.second.size() != 2) path_graph = false;
        }
        if (!path_graph || endpoints.size() != 2) continue;
        std::vector<int> path;
        int previous = -1, v = endpoints[0];
        while (v >= 0 && path.size() <= local.size()) {
            path.push_back(v);
            int next = -1;
            for (int u : adjacent[v]) if (u != previous) next = u;
            previous = v; v = next;
        }
        if (v < 0 && path.size() == local.size()) best_path = std::move(path);
    }
    // 未识别到路径时保持 -1，后续使用无位置轴的分核候选。
    std::vector<double> coord(g.ops.size(), -1), tensor_coord(g.tensors.size(), -1);
    for (int i = 0; i < static_cast<int>(best_path.size()); ++i) tensor_coord[best_path[i]] = i;
    // 先向输入预处理节点反传位置。
    for (auto it = g.topo.rbegin(); it != g.topo.rend(); ++it) {
        const int v = *it;
        if (level[v]) continue;
        std::vector<double> values;
        for (int t : g.ops[v].outputs) if (tensor_coord[t] >= 0) values.push_back(tensor_coord[t]);
        coord[v] = median_position(std::move(values));
        if (coord[v] >= 0) for (int t : g.ops[v].inputs) {
            if (tensor_coord[t] < 0) tensor_coord[t] = coord[v];
        }
    }
    // 只保留路径锚点，再按输入位置中位数向后传播。
    std::fill(tensor_coord.begin(), tensor_coord.end(), -1);
    for (int i = 0; i < static_cast<int>(best_path.size()); ++i) tensor_coord[best_path[i]] = i;
    for (int v : g.topo) {
        if (coord[v] < 0) {
            std::vector<double> values;
            for (int t : g.ops[v].inputs) {
                if (tensor_coord[t] >= 0) { values.push_back(tensor_coord[t]); continue; }
                std::vector<double> writers;
                for (int p : g.tensors[t].producers) if (!g.ops[p].copy && coord[p] >= 0) writers.push_back(coord[p]);
                if (!writers.empty()) values.push_back(median_position(std::move(writers)));
            }
            if (values.empty()) for (int p : g.pred[v]) if (coord[p] >= 0) values.push_back(coord[p]);
            coord[v] = median_position(std::move(values));
        }
    }
    // 共享预处理用消费者位置补齐，后续仍可迁移。
    for (auto it = g.topo.rbegin(); it != g.topo.rend(); ++it) if (coord[*it] < 0) {
        std::vector<double> values;
        for (int u : g.succ[*it]) if (coord[u] >= 0) values.push_back(coord[u]);
        coord[*it] = median_position(std::move(values));
    }
    return coord;
}


SchedulingGroups::SchedulingGroups(const Graph& g, const GraphAnalysis& analysis, bool use_compact)
    : graph(g), facts(analysis), of(g.ops.size(), -1), compact(use_compact) {
    // 仅在写者组能形成无环图时，提供连续执行的候选。
    std::vector<int> grouping = facts.co_group;
    std::vector<std::vector<int>> next(facts.co_groups.size());
    std::vector<int> group_degree(next.size());
    for (int v : g.work) for (int u : g.succ[v])
        if (grouping[v] != grouping[u]) next[grouping[v]].push_back(grouping[u]);
    for (auto& list : next) { unique_sort(list); for (int u : list) ++group_degree[u]; }
    std::vector<int> queue;
    for (int i = 0; i < static_cast<int>(group_degree.size()); ++i) if (!group_degree[i]) queue.push_back(i);
    for (std::size_t i = 0; i < queue.size(); ++i) for (int u : next[queue[i]]) if (!--group_degree[u]) queue.push_back(u);
    if (queue.size() != next.size()) for (int v : g.work) grouping[v] = v;
    std::map<int, int> groups;
    for (int v : g.work) {
        auto inserted = groups.emplace(grouping[v], static_cast<int>(units.size()));
        if (inserted.second) units.emplace_back();
        of[v] = inserted.first->second;
    }
    // 推断位置，汇总各单元的依赖和流水线负载。
    const auto coordinates = infer_locality_axis(g);
    for (int v : g.work) units[of[v]].ops.push_back(v);
    for (int v : g.work) {
        auto& unit = units[of[v]];
        unit.work[g.ops[v].pipe] += std::max<Cost>(1, g.ops[v].cycles);
        for (int u : g.succ[v]) if (of[v] != of[u]) unit.succ.push_back(of[u]);
        unit.inputs.insert(unit.inputs.end(), g.ops[v].inputs.begin(), g.ops[v].inputs.end());
        unit.outputs.insert(unit.outputs.end(), g.ops[v].outputs.begin(), g.ops[v].outputs.end());
    }
    for (int a = 0; a < static_cast<int>(units.size()); ++a) {
        auto& u = units[a];
        unique_sort(u.succ); unique_sort(u.inputs); unique_sort(u.outputs);
        u.uses = u.inputs; u.uses.insert(u.uses.end(), u.outputs.begin(), u.outputs.end()); unique_sort(u.uses);
        for (int b : u.succ) units[b].pred.push_back(a);
        std::vector<double> coords;
        for (int v : u.ops) if (coordinates[v] >= 0) coords.push_back(coordinates[v]);
        u.coordinate = median_position(std::move(coords)); spatial = spatial || u.coordinate >= 0;
        u.pipe = static_cast<int>(std::max_element(u.work.begin(), u.work.end()) - u.work.begin());
    }
    std::vector<int> degree(units.size());
    std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
    for (int v = 0; v < static_cast<int>(units.size()); ++v) {
        degree[v] = static_cast<int>(units[v].pred.size()); if (!degree[v]) ready.push(v);
    }
    while (!ready.empty()) {
        const int v = ready.top(); ready.pop(); topo.push_back(v);
        for (int u : units[v].succ) {
            units[u].depth = std::max(units[u].depth, units[v].depth + 1);
            if (!--degree[u]) ready.push(u);
        }
    }
    // 前者仅有这个后继、后者仅有这个前驱时，串成迁移候选链。
    for (int v : topo) {
        if (units[v].pred.size() == 1 && units[units[v].pred[0]].succ.size() == 1) continue;
        std::vector<int> chain{v};
        int u = v;
        while (units[u].succ.size() == 1 && units[units[u].succ[0]].pred.size() == 1) {
            u = units[u].succ[0]; chain.push_back(u);
        }
        chains.push_back(std::move(chain));
    }
    // 记录每个张量涉及的组，迁组时只重算关联张量的搬运费用。
    for (const auto& t : g.tensors) {
        TensorLinks net; net.service = transfer_cycles(t.bytes);
        for (int v : t.producers) if (!g.ops[v].copy) net.producers.push_back(of[v]);
        for (int v : t.consumers) {
            if (!g.ops[v].copy) net.consumers.push_back(of[v]);
            if (g.ops[v].copy == 2) net.output = true;
        }
        add_net(std::move(net));
    }
    for (std::size_t i = 0; i < g.direct.size(); ++i) {
        const auto e = g.direct[i];
        if (!g.ops[e.first].copy && !g.ops[e.second].copy)
            add_net(TensorLinks{{of[e.first]}, {of[e.second]}, transfer_cycles(g.direct_bytes[i]), false});
    }
    prepare_order(use_compact);
}

void SchedulingGroups::prepare_order(bool use_compact) {
    compact = use_compact;
    const auto original = compact ? dependency_order(graph, true) : reference_order_with_copies(graph);
    for (auto& unit : units) unit.ops.clear();
    for (int v : original) units[of[v]].ops.push_back(v);
    dfs.clear();
    std::vector<int> degree(units.size());
    std::vector<int> priority(units.size(), INT32_MAX);
    for (int i = 0; i < static_cast<int>(original.size()); ++i) priority[of[original[i]]] = std::min(priority[of[original[i]]], i);
    using Entry = std::pair<int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> ordered;
    for (int v : topo) { degree[v] = static_cast<int>(units[v].pred.size()); if (!degree[v]) ordered.emplace(priority[v], v); }
    while (!ordered.empty()) {
        const int v = ordered.top().second; ordered.pop(); dfs.push_back(v);
        for (int u : units[v].succ) if (!--degree[u]) ordered.emplace(priority[u], u);
    }
    if (compact) {
        dfs.clear();
        std::vector<int> stack;
        std::vector<char> visited(units.size());
        const auto less = [&](int a, int b) { return units[a].depth != units[b].depth ? units[a].depth < units[b].depth : a > b; };
        for (int v : topo) if (units[v].succ.empty()) stack.push_back(v);
        std::sort(stack.begin(), stack.end(), less);
        while (!stack.empty()) {
            const int v = stack.back();
            if (visited[v]) { stack.pop_back(); continue; }
            std::vector<int> pending;
            for (int p : units[v].pred) if (!visited[p]) pending.push_back(p);
            if (pending.empty()) { visited[v] = 1; dfs.push_back(v); stack.pop_back(); }
            else { std::sort(pending.begin(), pending.end(), less); stack.insert(stack.end(), pending.begin(), pending.end()); }
        }
    }
}

void SchedulingGroups::add_net(TensorLinks net) {
    unique_sort(net.producers); unique_sort(net.consumers);
    auto touch = net.producers; touch.insert(touch.end(), net.consumers.begin(), net.consumers.end()); unique_sort(touch);
    for (int v : touch) units[v].nets.push_back(static_cast<int>(nets.size()));
    nets.push_back(std::move(net));
}

std::vector<int> SchedulingGroups::normalize_owners(std::vector<int> owner) const {
    for (const auto& group : facts.co_groups) {
        const int core = owner[of[group.front()]];
        for (int op : group) owner[of[op]] = core;
    }
    return owner;
}

std::vector<std::pair<int, int>> SchedulingGroups::close_changes(const std::vector<std::pair<int, int>>& changes) const {
    std::map<int, int> destination;
    for (const auto& change : changes) for (int op : units[change.first].ops) {
        const int group = facts.co_group[op];
        auto item = destination.emplace(group, change.second);
        if (!item.second && item.first->second != change.second) return {};
    }
    std::vector<std::pair<int, int>> result;
    for (const auto& item : destination) for (int op : facts.co_groups[item.first]) result.emplace_back(of[op], item.second);
    unique_sort(result);
    return result;
}

Plan SchedulingGroups::project(const std::vector<int>& owner, const std::vector<int>& order, int cores) const {
    const auto fixed = normalize_owners(owner);
    Plan p; p.orders.resize(cores);
    for (int u : order) for (int v : units[u].ops) p.orders[fixed[u]].push_back(v);
    return p;
}

std::vector<int> SchedulingGroups::owners(const Plan& plan) const {
    std::vector<int> owner(units.size());
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c)
        for (int op : plan.orders[c]) owner[of[op]] = c;
    return owner;
}

} // npu 命名空间
