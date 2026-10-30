#include "conv.hpp"
#include "search.hpp"

#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#ifdef NPU_SEARCH_TRACE
#include <iostream>
#endif

namespace npu {
namespace {

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int a) { while (a != parent[a]) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
    void join(int a, int b) { a = find(a); b = find(b); if (a != b) parent[std::max(a, b)] = std::min(a, b); }
};

double median(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return (v[(v.size() - 1) / 2] + v[v.size() / 2]) * 0.5;
}

// 含 COPY 求默认遍历序，再保留可分配的计算节点。
std::vector<int> original_order(const Graph& g) {
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
std::vector<double> locality_axis(const Graph& g) {
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
        coord[v] = median(std::move(values));
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
                if (!writers.empty()) values.push_back(median(std::move(writers)));
            }
            if (values.empty()) for (int p : g.pred[v]) if (coord[p] >= 0) values.push_back(coord[p]);
            coord[v] = median(std::move(values));
        }
    }
    // 共享预处理用消费者位置补齐，后续仍可迁移。
    for (auto it = g.topo.rbegin(); it != g.topo.rend(); ++it) if (coord[*it] < 0) {
        std::vector<double> values;
        for (int u : g.succ[*it]) if (coord[u] >= 0) values.push_back(coord[u]);
        coord[*it] = median(std::move(values));
    }
    return coord;
}

// 绑定同核、连续排列的操作组；输出时仍逐 op 写子图。
struct Unit {
    std::vector<int> ops, pred, succ, inputs, outputs, uses, nets;
    std::array<Cost, kPipeCount> work{};
    int depth = 0, pipe = 0;
    double coordinate = -1;
};
// 张量或直接边对应的生产组、消费组和搬运服务时间。
struct Net {
    std::vector<int> producers, consumers;
    Cost service = 0;
    bool output = false;
};

// 图预处理结果：基础分组、依赖、位置和搬运关系。
struct Regions {
    const Graph& graph;
    std::vector<Unit> units;
    std::vector<int> of, topo, dfs;
    std::vector<Net> nets;
    std::vector<std::vector<int>> chains;
    bool spatial = false;
    bool compact;

    explicit Regions(const Graph& g, bool use_compact = false) : graph(g), of(g.ops.size(), -1), compact(use_compact) {
        // 同一张量 T 若由 P、Q 写入，就把 P、Q 合组并绑定同核。
        UnionFind uf(static_cast<int>(g.ops.size()));
        for (const auto& t : g.tensors) {
            int first = -1;
            for (int p : t.producers) if (!g.ops[p].copy) {
                if (first < 0) first = p; else uf.join(first, p);
            }
        }
        // 收缩分组可能引入环；合并强连通分量，得到无环单元图。
        for (;;) {
            std::vector<std::vector<int>> adj(g.ops.size()), rev(g.ops.size());
            for (int v : g.work) for (int u : g.succ[v]) {
                const int a = uf.find(v), b = uf.find(u);
                if (a != b) { adj[a].push_back(b); rev[b].push_back(a); }
            }
            std::vector<char> seen(g.ops.size());
            std::vector<int> finish;
            for (int root : g.work) if (uf.find(root) == root && !seen[root]) {
                std::vector<std::pair<int, std::size_t>> stack{{root, 0}}; seen[root] = 1;
                while (!stack.empty()) {
                    auto& frame = stack.back();
                    if (frame.second == adj[frame.first].size()) { finish.push_back(frame.first); stack.pop_back(); continue; }
                    const int u = adj[frame.first][frame.second++];
                    if (!seen[u]) { seen[u] = 1; stack.emplace_back(u, 0); }
                }
            }
            std::fill(seen.begin(), seen.end(), 0);
            bool changed = false;
            for (auto it = finish.rbegin(); it != finish.rend(); ++it) if (!seen[*it]) {
                std::vector<int> stack{*it}; seen[*it] = 1;
                while (!stack.empty()) {
                    const int v = stack.back(); stack.pop_back();
                    if (uf.find(v) != uf.find(*it)) { uf.join(v, *it); changed = true; }
                    for (int u : rev[v]) if (!seen[u]) { seen[u] = 1; stack.push_back(u); }
                }
            }
            if (!changed) break;
        }
        std::map<int, int> groups;
        for (int v : g.work) {
            const int root = uf.find(v);
            auto inserted = groups.emplace(root, static_cast<int>(units.size()));
            if (inserted.second) units.emplace_back();
            of[v] = inserted.first->second;
        }
        // 推断位置，汇总各单元的依赖和流水线负载。
        const auto coordinates = locality_axis(g);
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
            u.coordinate = median(std::move(coords)); spatial = spatial || u.coordinate >= 0;
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
            Net net; net.service = transfer_cycles(t.bytes);
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
                add_net(Net{{of[e.first]}, {of[e.second]}, transfer_cycles(g.direct_bytes[i]), false});
        }
        prepare_order(use_compact);
    }

    // 两套搜索只切换参考顺序，复用分组、位置和搬运关系。
    void prepare_order(bool use_compact) {
        compact = use_compact;
        const auto original = compact ? dependency_order(graph, true) : original_order(graph);
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

    void add_net(Net net) {
        unique_sort(net.producers); unique_sort(net.consumers);
        auto touch = net.producers; touch.insert(touch.end(), net.consumers.begin(), net.consumers.end()); unique_sort(touch);
        for (int v : touch) units[v].nets.push_back(static_cast<int>(nets.size()));
        nets.push_back(std::move(net));
    }

    Plan project(const std::vector<int>& owner, const std::vector<int>& order, int cores) const {
        Plan p; p.orders.resize(cores);
        for (int u : order) for (int v : units[u].ops) p.orders[owner[u]].push_back(v);
        return p;
    }

    std::vector<int> owners(const Plan& plan) const {
        std::vector<int> owner(units.size());
        for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c)
            for (int op : plan.orders[c]) owner[of[op]] = c;
        return owner;
    }
};

using Changes = std::vector<std::pair<int, int>>;

double band_step(const Regions& r, int cores) {
    std::map<int, std::vector<double>> layers;
    for (int v : r.topo) if (r.units[v].coordinate >= 0) layers[r.units[v].depth].push_back(r.units[v].coordinate);
    double step = 1;
    for (auto& item : layers) {
        auto& positions = item.second; unique_sort(positions);
        if (positions.size() < static_cast<std::size_t>(2 * cores)) continue;
        std::vector<double> gaps;
        for (std::size_t i = 1; i < positions.size(); ++i) gaps.push_back(positions[i] - positions[i - 1]);
        const double spacing = median(std::move(gaps));
        // 仅用近似等间距的位置确定粗条带边界。
        int regular = 0;
        for (std::size_t i = 1; i < positions.size(); ++i)
            regular += std::abs((positions[i] - positions[i - 1]) - spacing) < 0.01 * spacing;
        if (regular * 4 >= static_cast<int>(positions.size() - 1) * 3) step = std::max(step, spacing);
    }
    return step;
}

// 迁移时只更新受影响单元及关联张量的负载和通信量。
struct Placement {
    const Regions& r;
    std::vector<int> owner;
    std::vector<std::array<int, 5>> producers, consumers;
    std::vector<std::array<Cost, kPipeCount>> load;
    Cost ddr = 0;
    Placement(const Regions& regions, std::vector<int> assignment, int cores)
        : r(regions), owner(std::move(assignment)), producers(r.nets.size()), consumers(r.nets.size()), load(cores) {
        for (int v : r.topo) for (int p = 0; p < kPipeCount; ++p) load[owner[v]][p] += r.units[v].work[p];
        for (int t = 0; t < static_cast<int>(r.nets.size()); ++t) {
            for (int v : r.nets[t].producers) ++producers[t][owner[v]];
            for (int v : r.nets[t].consumers) ++consumers[t][owner[v]];
            ddr += cost(t);
        }
    }
    Cost cost(int t) const {
        int p = 0, c = 0, overlap = 0;
        for (int i = 0; i < static_cast<int>(load.size()); ++i) {
            p += producers[t][i] > 0; c += consumers[t][i] > 0;
            overlap += producers[t][i] > 0 && consumers[t][i] > 0;
        }
        int copies = 2 * (p * c - overlap);
        if (!p) copies += c;
        else if (r.nets[t].output || !c) copies += p;
        return copies * r.nets[t].service;
    }
    Cost bound() const {
        Cost b = ddr;
        for (const auto& core : load) for (Cost w : core) b = std::max(b, w);
        return b;
    }
    double estimate() const {
        Cost total_peak = 0;
        for (const auto& core : load) total_peak = std::max(total_peak, std::accumulate(core.begin(), core.end(), Cost{0}));
        return static_cast<double>(bound()) + 0.15 * static_cast<double>(ddr) + 0.05 * static_cast<double>(total_peak);
    }
    Changes apply(const Changes& changes) {
        Changes undo;
        for (const auto& change : changes) {
            const int v = change.first, a = owner[v], b = change.second;
            if (a == b) continue;
            undo.emplace_back(v, a);
            for (int t : r.units[v].nets) ddr -= cost(t);
            for (int p = 0; p < kPipeCount; ++p) { load[a][p] -= r.units[v].work[p]; load[b][p] += r.units[v].work[p]; }
            for (int t : r.units[v].nets) {
                const auto& net = r.nets[t];
                if (std::binary_search(net.producers.begin(), net.producers.end(), v)) { --producers[t][a]; ++producers[t][b]; }
                if (std::binary_search(net.consumers.begin(), net.consumers.end(), v)) { --consumers[t][a]; ++consumers[t][b]; }
                ddr += cost(t);
            }
            owner[v] = b;
        }
        std::reverse(undo.begin(), undo.end());
        return undo;
    }
};

// 前驱都已排入顺序的组才可选，保持全局无环。
// 时间和内存状态仅作排序估计，完整方案再由模型评估。
std::vector<int> reorder(const Regions& r, const std::vector<int>& owner, int cores,
                         const std::vector<int>& reference, int window, double matrix_bias, double pressure_weight) {
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
    std::vector<int> result; result.reserve(n);
    while (static_cast<int>(result.size()) < n) {
        int best = -1;
        Cost best_start = 0;
        std::tuple<double, double, int, int> best_key;
        for (const auto& queue : ready) {
            int inspected = 0;
            for (const auto& item : queue) {
                const int v = item.second, c = owner[v];
                // 每队列最多看 12/64 组，并限制偏离原顺序的距离。
                if (inspected++ == (r.compact ? 12 : 64) || (inspected > 1 && item.first >= cursor[c] + window)) break;
                Cost start = last_start[c];
                for (int p = 0; p < kPipeCount; ++p) if (r.units[v].work[p]) start = std::max(start, pipe_end[c][p]);
                for (int p : r.units[v].pred) start = std::max(start, finish[p] + (owner[p] == c ? 0 : kCrossCoreDelay));
                std::array<Cost, 2> alloc{}, freed{};
                for (int t : r.units[v].uses) {
                    const auto& tensor = r.graph.tensors[t]; const int storage = tensor.storage == 0 ? 0 : 1;
                    if (!resident[t * cores + c]) alloc[storage] += tensor.bytes;
                    if (remaining[t * cores + c] == 1) freed[storage] += tensor.bytes;
                }
                double pressure = 0, delta = 0;
                for (int p = 0; p < 2; ++p) {
                    pressure += std::max(0.0, static_cast<double>(used[c][p] + alloc[p]) / (0.85 * kCapacity[p]) - 1);
                    delta += static_cast<double>(alloc[p] - freed[p]) / kCapacity[p];
                }
                const auto key = std::make_tuple(static_cast<double>(start) + matrix_bias * (r.units[v].pipe != 2) + pressure_weight * pressure,
                                                 delta, rank[v] - cursor[c], v);
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
        for (int t : r.units[best].uses) {
            const auto& tensor = r.graph.tensors[t]; const int storage = tensor.storage == 0 ? 0 : 1, index = t * cores + c;
            if (!resident[index]) { resident[index] = 1; used[c][storage] += tensor.bytes; }
            if (!--remaining[index]) { resident[index] = 0; used[c][storage] -= tensor.bytes; }
        }
        done[best] = 1; result.push_back(best);
        while (cursor[c] < static_cast<int>(base[c].size()) && done[base[c][cursor[c]]]) ++cursor[c];
        for (int u : r.units[best].succ) if (!--degree[u]) ready[owner[u] * kPipeCount + r.units[u].pipe].emplace(rank[u], u);
    }
    return result;
}

// 将生产者移近同核消费者，缩短输出驻留。
// 仅跨越无依赖操作，比较输出收益与输入驻留代价。
Plan just_in_time(const Regions& r, const Plan& base, double factor, bool expanding_only, bool single_successor_only,
                 int only_depth = -1) {
    Plan result = base;
    const auto owner = r.owners(base);
    const int cores = static_cast<int>(base.orders.size());
    std::vector<std::vector<int>> orders(cores), users(r.graph.tensors.size());
    for (int c = 0; c < cores; ++c) for (int op : base.orders[c]) {
        const int v = r.of[op];
        if (orders[c].empty() || orders[c].back() != v) orders[c].push_back(v);
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

// 共用评估与预算，额外保留至多四种分核供局部搜索。
struct GeneralSearch : PlanSearchState {
    const Regions& r;
    std::vector<Candidate> beam;

    GeneralSearch(const Regions& regions, int n)
        : PlanSearchState(regions.graph, n,
              static_cast<int>(std::min<std::size_t>(144, std::max<std::size_t>(20,
                  1800000 / std::max<std::size_t>(1, regions.graph.work.size())))), StopRule::TimeBound),
          r(regions) {}

    void consider(Plan p, const char* label) {
        auto score = evaluate_candidate(p, label);
        if (!score.valid) return;
        Candidate candidate{std::move(p), std::move(score)};
        // 每种分核保留最好的顺序。
        const auto owner = r.owners(candidate.plan);
        bool present = false;
        for (auto& old : beam) if (r.owners(old.plan) == owner) {
            if (better(candidate, old)) old = candidate;
            present = true; break;
        }
        if (!present) beam.push_back(std::move(candidate));
        std::stable_sort(beam.begin(), beam.end(), better);
        if (beam.size() > 4) beam.resize(4);
    }
};

// 结合已测开始时间和原有依赖，生成合法参考顺序。
std::vector<int> reference_order(const Regions& r, const Candidate& candidate) {
    std::vector<std::vector<int>> next(r.units.size());
    for (int v : r.topo) next[v] = r.units[v].succ;
    for (const auto& order : candidate.plan.orders) {
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
    for (int v : r.topo) if (!degree[v]) ready.emplace(candidate.score.op_start[r.units[v].ops[0]], v);
    std::vector<int> result;
    while (!ready.empty()) {
        const int v = ready.top().second; ready.pop(); result.push_back(v);
        for (int u : next[v]) if (!--degree[u]) ready.emplace(candidate.score.op_start[r.units[u].ops[0]], u);
    }
    return result.size() == r.units.size() ? result : r.dfs;
}

// 只生成核归属，按粗估筛选后交给排序函数。
std::vector<std::vector<int>> make_initial_placements(const Regions& r, int cores) {
    const int count = static_cast<int>(r.units.size());
    std::vector<std::vector<int>> assignments;
    // 无依赖连接的区域分别整组分核。
    UnionFind connected(count);
    for (int v : r.topo) for (int u : r.units[v].succ) connected.join(v, u);
    std::map<int, std::vector<int>> components;
    for (int v : r.topo) components[connected.find(v)].push_back(v);
    if (components.size() > 1) {
        std::vector<std::pair<Cost, int>> ranked;
        for (const auto& component : components) {
            Cost work = 0;
            for (int v : component.second) work += std::accumulate(r.units[v].work.begin(), r.units[v].work.end(), Cost{0});
            ranked.emplace_back(-work, component.first);
        }
        std::sort(ranked.begin(), ranked.end());
        std::vector<Cost> load(cores);
        std::vector<int> owner(count);
        for (const auto& item : ranked) {
            const int c = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
            for (int v : components[item.second]) owner[v] = c;
            load[c] -= item.first;
        }
        assignments.push_back(std::move(owner));
    }
    // 有位置编号时按编号排序，否则沿依赖遍历。
    std::vector<int> sorted = r.dfs;
    if (r.spatial) std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        return r.units[a].coordinate < r.units[b].coordinate;
    });
    Cost total = 0;
    for (int v : sorted) total += std::accumulate(r.units[v].work.begin(), r.units[v].work.end(), Cost{0});
    double maximum = 0;
    for (int v : sorted) maximum = std::max(maximum, r.units[v].coordinate);
    // 比较计算量均衡、位置等分及边界偏移的分核。
    for (int q = cores; q >= 2; --q) {
        for (int style = 0; style < (r.spatial ? 3 : 1); ++style) {
            std::vector<int> owner(count);
            Cost accumulated = 0;
            for (int v : sorted) {
                const Cost w = std::accumulate(r.units[v].work.begin(), r.units[v].work.end(), Cost{0});
                double fraction = static_cast<double>(accumulated + w / 2) / std::max<Cost>(1, total);
                if (style) fraction = (r.units[v].coordinate + 0.5) / (maximum + 1);
                // 小幅移动边界，比较邻近分核方案。
                if (style == 2) fraction += 0.25 / (maximum + 1);
                owner[v] = std::max(0, std::min(q - 1, static_cast<int>(fraction * q)));
                accumulated += w;
            }
            assignments.push_back(std::move(owner));
        }
        // 宽窗口模式额外尝试与粗层位置对齐的条带。
        if (r.spatial && !r.compact) {
            const double step = band_step(r, q);
            const int cells = std::max(q, static_cast<int>(std::ceil((maximum + 1) / step)));
            std::vector<int> widths(q, cells / q);
            for (int i = 0; i < cells % q; ++i) ++widths[q - 1 - i];
            do {
                std::vector<int> ends(q); std::partial_sum(widths.begin(), widths.end(), ends.begin());
                std::vector<int> owner(count);
                for (int v : r.topo) {
                    const int cell = std::max(0, static_cast<int>(r.units[v].coordinate / step));
                    owner[v] = std::min(q - 1, static_cast<int>(std::upper_bound(ends.begin(), ends.end(), cell) - ends.begin()));
                    // 小而共享的根单元固定到核 0。
                    if (r.units[v].pred.empty() && r.units[v].succ.size() >= static_cast<std::size_t>(4 * cores) &&
                        std::accumulate(r.units[v].work.begin(), r.units[v].work.end(), Cost{0}) <= kCrossCoreDelay / 8) owner[v] = 0;
                }
                assignments.push_back(std::move(owner));
            } while (std::next_permutation(widths.begin(), widths.end()));
        }
    }
    // 无位置轴时，按一对一依赖链的计算量从大到小均衡分核。
    if (!r.spatial) {
        std::vector<int> chain_ids(r.chains.size()); std::iota(chain_ids.begin(), chain_ids.end(), 0);
        std::vector<Cost> weights(r.chains.size());
        for (int i : chain_ids) for (int v : r.chains[i]) weights[i] += std::accumulate(r.units[v].work.begin(), r.units[v].work.end(), Cost{0});
        std::stable_sort(chain_ids.begin(), chain_ids.end(), [&](int a, int b) { return weights[a] > weights[b]; });
        std::vector<Cost> load(cores);
        std::vector<int> owner(count);
        for (int i : chain_ids) {
            const int c = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
            for (int v : r.chains[i]) owner[v] = c;
            load[c] += weights[i];
        }
        assignments.push_back(std::move(owner));
    }
    // 去重并按负载、搬运粗估筛选，最多保留 12 种分核。
    std::sort(assignments.begin(), assignments.end());
    assignments.erase(std::unique(assignments.begin(), assignments.end()), assignments.end());
    // 每种分核只计算一次粗估费用。
    struct Ranked { double estimate; std::vector<int> owner; };
    std::vector<Ranked> ranked;
    for (auto& owner : assignments) {
        const double estimate = Placement(r, owner, cores).estimate();
        ranked.push_back(Ranked{estimate, std::move(owner)});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.estimate < b.estimate;
    });
    if (ranked.size() > 12) ranked.resize(12);
    assignments.clear();
    for (auto& item : ranked) assignments.push_back(std::move(item.owner));
    return assignments;
}

// 保留现有两步延后顺序，先处理输出膨胀的组，再处理其中后继唯一的组。
Plan delay_expanding_producers(const Regions& r, Plan p) {
    p = just_in_time(r, p, 1, true, false);
    return just_in_time(r, p, 1, true, true);
}

// 固定分核，比较参考顺序、就绪重排和延后生产者。
void try_initial_orders(GeneralSearch& search, const std::vector<int>& owner, double bias) {
    const auto& r = search.r;
    if (search.should_stop()) return;
    search.consider(r.project(owner, r.dfs, search.cores), "placement-dfs");
    for (double preference : {0.0, bias}) {
        if (search.should_stop()) return;
        const auto order = reorder(r, owner, search.cores, r.dfs, 4096, preference, 0);
        Plan p = r.project(owner, order, search.cores);
        search.consider(p, "placement-list");
        if (search.should_stop()) return;
        search.consider(delay_expanding_producers(r, std::move(p)), "placement-jit");
    }
}

void try_initial_plans(GeneralSearch& search) {
    const auto& r = search.r;
    const auto assignments = make_initial_placements(r, search.cores);
    std::vector<Cost> durations;
    for (const auto& unit : r.units) if (unit.work[2] > 1) durations.push_back(unit.work[2]);
    std::sort(durations.begin(), durations.end());
    const double bias = durations.empty() ? 0 : (r.compact ? 2.0 : 2.5) * durations[durations.size() / 2];
#ifdef NPU_SEARCH_TRACE
    std::cerr << "seed axis=" << r.spatial << " step=" << band_step(r, search.cores) << " bias=" << bias << '\n';
#endif
    for (const auto& owner : assignments) {
        if (search.should_stop()) break;
        try_initial_orders(search, owner, bias);
    }
}

// 固定核归属，尝试缩短驻留和重排就绪单元。
void improve_orders(GeneralSearch& search, int budget) {
    const int end = std::min(search.max_evaluations, search.evaluated_count + budget);
    auto seeds = search.beam;
    if (seeds.size() > 3) seeds.resize(3);
    for (const auto& seed : seeds) {
        const auto owner = search.r.owners(seed.plan);
        const auto reference = reference_order(search.r, seed);
        for (double factor : {1.0, 0.5, 2.0}) {
            if (search.evaluated_count >= end || search.should_stop()) return;
            auto p = just_in_time(search.r, seed.plan, factor, false, false);
            search.consider(std::move(p), "live-area");
        }
        // 单独尝试延后换出量最大的几个层。
        std::map<int, Cost> pressure;
        if (!search.r.compact) for (int v : search.r.topo) for (int t : search.r.units[v].outputs)
            pressure[search.r.units[v].depth] += seed.score.tensor_spill_bytes[t];
        std::vector<std::pair<Cost, int>> stages;
        for (const auto& item : pressure) if (item.second) stages.emplace_back(-item.second, item.first);
        std::sort(stages.begin(), stages.end());
        if (stages.size() > 3) stages.resize(3);
        for (const auto& stage : stages) {
            if (search.evaluated_count >= end || search.should_stop()) return;
            auto p = just_in_time(search.r, seed.plan, 0.5, false, false, stage.second);
            search.consider(std::move(p), "spill-stage-jit");
        }
        for (int window : {64, 512}) {
            if (search.evaluated_count >= end || search.should_stop()) return;
            const auto order = reorder(search.r, owner, search.cores, reference, window, 0, 4.0 * kCrossCoreDelay);
            auto p = search.r.project(owner, order, search.cores);
            p = just_in_time(search.r, p, 1, false, false);
            search.consider(std::move(p), "memory-list");
        }
    }
}

struct Move { Changes changes; double estimate; int kind; };

// 生成迁移、拆分、交换和条带调整候选，先做静态筛选。
std::vector<Move> propose_moves(const Regions& r, const Candidate& seed, int cores, Cost incumbent) {
    Placement state(r, r.owners(seed.plan), cores);
    const auto owner = state.owner;
    const double baseline = state.estimate();
    std::array<std::vector<Move>, 5> pools;
    const auto offer = [&](Changes changes, int kind) {
        changes.erase(std::remove_if(changes.begin(), changes.end(), [&](const auto& x) { return owner[x.first] == x.second; }), changes.end());
        if (changes.empty()) return;
        std::sort(changes.begin(), changes.end());
        changes.erase(std::unique(changes.begin(), changes.end()), changes.end());
        auto undo = state.apply(changes);
        if (state.bound() <= incumbent) {
            auto& pool = pools[kind];
            pool.push_back(Move{std::move(changes), state.estimate() - baseline, kind});
            if (pool.size() > 40) {
                std::stable_sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) { return std::tie(a.estimate, a.changes) < std::tie(b.estimate, b.changes); });
                pool.resize(16);
            }
        }
        state.apply(undo);
    };
    // 尝试整条一对一依赖链迁移或只迁移前半、后半；基础单元不拆。
    for (const auto& chain : r.chains) {
        std::set<int> destinations;
        for (int v : chain) {
            for (int p : r.units[v].pred) destinations.insert(owner[p]);
            for (int u : r.units[v].succ) destinations.insert(owner[u]);
        }
        const int light = static_cast<int>(std::min_element(seed.score.core_end.begin(), seed.score.core_end.end()) - seed.score.core_end.begin());
        destinations.insert(light);
        for (int c : destinations) {
            Changes changes;
            for (int v : chain) changes.emplace_back(v, c);
            offer(std::move(changes), 0);
            if (chain.size() > 1) for (int side = 0; side < 2; ++side) {
                changes.clear();
                for (std::size_t i = side ? chain.size() / 2 : 0; i < (side ? chain.size() : chain.size() / 2); ++i)
                    changes.emplace_back(chain[i], c);
                offer(std::move(changes), 1);
            }
        }
    }
    // 优先检查完成较晚、换出较多的跨核边界单元。
    std::vector<std::pair<double, int>> boundary;
    for (int v : r.topo) {
        bool cut = false;
        for (int p : r.units[v].pred) cut = cut || owner[p] != owner[v];
        for (int u : r.units[v].succ) cut = cut || owner[u] != owner[v];
        if (!cut) continue;
        double priority = 0;
        for (int t : r.units[v].uses) priority += seed.score.tensor_spill_bytes[t] / static_cast<double>(kBandwidth);
        for (int op : r.units[v].ops) priority = std::max(priority, static_cast<double>(seed.score.op_end[op]));
        boundary.emplace_back(-priority, v);
    }
    std::sort(boundary.begin(), boundary.end());
    if (boundary.size() > 192) boundary.resize(192);
    for (const auto& item : boundary) {
        const int v = item.second;
        for (int u : r.units[v].succ) if (owner[u] != owner[v]) {
            offer({{v, owner[u]}}, 1); offer({{u, owner[v]}}, 1);
            offer({{v, owner[u]}, {u, owner[v]}}, 3);
            // 尝试生产者和其直接消费者一起迁到同核。
            if (r.units[v].succ.size() <= 8) {
                Changes changes{{v, owner[u]}};
                for (int w : r.units[v].succ) changes.emplace_back(w, owner[u]);
                offer(std::move(changes), 2);
            }
        }
    }
    // 将边界附近一条薄带整体迁核，保留带内的数据复用。
    if (r.spatial) {
        double step = band_step(r, cores);
        if (r.compact) {
            std::vector<double> outputs;
            for (int v : r.topo) if (r.units[v].succ.empty() && r.units[v].coordinate >= 0) outputs.push_back(r.units[v].coordinate);
            unique_sort(outputs);
            std::vector<double> gaps;
            for (std::size_t i = 1; i < outputs.size(); ++i) gaps.push_back(outputs[i] - outputs[i - 1]);
            step = gaps.empty() ? 1.0 : median(std::move(gaps));
        }
        for (int a = 0; a < cores; ++a) for (int b = 0; b < cores; ++b) if (a != b) {
            double center = 0; int count = 0;
            for (int v : r.topo) if (owner[v] == b && r.units[v].coordinate >= 0) { center += r.units[v].coordinate; ++count; }
            if (!count) continue;
            center /= count;
            std::vector<int> near;
            for (int v : r.topo) if (owner[v] == a && r.units[v].coordinate >= 0) near.push_back(v);
            if (near.empty()) continue;
            std::stable_sort(near.begin(), near.end(), [&](int x, int y) { return std::abs(r.units[x].coordinate - center) < std::abs(r.units[y].coordinate - center); });
            const double distance = std::abs(r.units[near[0]].coordinate - center);
            for (double width : {0.5 * step, step, 2 * step}) {
                Changes changes;
                for (int v : near) {
                    if (std::abs(r.units[v].coordinate - center) >= distance + width) break;
                    changes.emplace_back(v, b);
                }
                offer(std::move(changes), 4);
            }
        }
    }
    std::vector<Move> result;
    std::set<Changes> seen;
    for (auto& pool : pools) {
        std::stable_sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) { return std::tie(a.estimate, a.changes) < std::tie(b.estimate, b.changes); });
        int selected = 0;
        for (auto& move : pool) if (seen.insert(move.changes).second) {
            result.push_back(std::move(move));
            if (++selected == 3) break;
        }
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return std::tie(a.estimate, a.kind, a.changes) < std::tie(b.estimate, b.kind, b.changes); });
    return result;
}

// 联合修改核归属和执行顺序，最多进行三轮局部搜索。
void refine_placements(GeneralSearch& search) {
    const int end = search.max_evaluations - (search.r.compact ? 0 : 12);
    for (int round = 0; round < 3 && !search.should_stop() && search.evaluated_count < end; ++round) {
        auto seeds = search.beam;
        if (seeds.size() > 2) seeds.resize(2);
        const auto previous = fingerprint(search.best.plan);
        for (const auto& seed : seeds) {
            const auto moves = propose_moves(search.r, seed, search.cores, search.best.score.makespan);
            const auto reference = reference_order(search.r, seed);
            for (const auto& move : moves) {
                if (search.should_stop() || search.evaluated_count >= end) return;
                auto owner = search.r.owners(seed.plan);
                for (const auto& change : move.changes) owner[change.first] = change.second;
                // 迁核后，先保留原有顺序并尝试延后生产者。
                auto p = search.r.project(owner, reference, search.cores);
                p = just_in_time(search.r, p, 1, false, false);
                search.consider(std::move(p), "move-preserve");
                // 再重排核内顺序，评估同一迁移的另一种执行方式。
                if (search.should_stop() || search.evaluated_count >= end) return;
                const auto order = reorder(search.r, owner, search.cores, reference, 512, 0, 0);
                p = search.r.project(owner, order, search.cores);
                p = delay_expanding_producers(search.r, std::move(p));
                search.consider(std::move(p), "move-reorder");
            }
        }
        improve_orders(search, 8);
        if (fingerprint(search.best.plan) == previous) break;
    }
}

Candidate run_search(const Regions& regions, int cores) {
    const auto& graph = regions.graph;
    // 初始化本轮搜索记录。
    GeneralSearch search(regions, cores);

    // 1. 全部组放核 0，分别按依赖遍历序和拓扑序生成方案。
    std::vector<int> single(regions.units.size());
    search.consider(regions.project(single, regions.dfs, cores), "single-dfs");
    search.consider(regions.project(single, regions.topo, cores), "single-topo");

    // 2. 生成并评估多核初始方案。
    if (!search.should_stop()) try_initial_plans(search);

    // 3. 保持核归属，先改进执行顺序。
    if (!search.should_stop()) improve_orders(search, 16);

    // 4. 从较好方案出发，尝试迁核并配合重排。
    if (!search.should_stop() && search.best.score.valid) refine_placements(search);

    // 5. 对改进后的分核再优化一次顺序。
    if (!search.should_stop()) improve_orders(search, 12);
    if (!search.best.score.valid) {
        // 都失败时，尝试原图单核顺序，放开组内连续排列的限制。
        Plan fallback; fallback.orders.resize(cores);
        fallback.orders[0] = dependency_order(graph, true);
        auto score = evaluate(graph, fallback);
        if (score.valid) return Candidate{std::move(fallback), std::move(score)};
    }
    return search.best;
}

} // 匿名命名空间

Plan solve_general(const Graph& graph, int cores) {
    // 第一套：选下一组时，每个就绪队列最多比较 12 组。
    Regions regions(graph, true);
    Candidate best = run_search(regions, cores);
    // 时间已达到下界时提前结束。
    if (best.score.valid && best.score.makespan == universal_lower_bound(graph, cores)) return best.plan;

    // 第二套：每队列最多比较 64 组，另试粗分段和按层延后。
    // run_search 新建搜索记录，两套评估次数预算互相独立。
    regions.prepare_order(false);
    Candidate other = run_search(regions, cores);
    // 两轮结果一起比较，返回较好的方案。
    return better(other, best) ? other.plan : best.plan;
}
} // npu 命名空间
