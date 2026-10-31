#include "search/candidate_generator.hpp"
#include "search/common.hpp"
namespace npu {
namespace search_detail {
CandidateGenerator::CandidateGenerator(const Problem &g, int n)
    : problem(g), cores(n), size(static_cast<int>(g.ops.size())), depth(size), atom(size), members(size),
      loads(size) {
    std::vector<int> degree(size);
    std::priority_queue<int, std::vector<int>, std::greater<int>> queue;
    for (int u : g.op_topo) {
        degree[u] = static_cast<int>(g.ops[u].preds.size());
        if (!degree[u]) {
            queue.push(u);
        }
    }
    while (!queue.empty()) {
        int u = queue.top();
        queue.pop();
        topo.push_back(u);
        for (int v : g.ops[u].succs) {
            depth[v] = std::max(depth[v], depth[u] + 1);
            if (!--degree[v]) {
                queue.push(v);
            }
        }
    }
    UnionFind d(size);
    for (const auto &t : g.tensors) {
        for (std::size_t i = 1; i < t.producers.size(); ++i) {
            d.join(t.producers.front(), t.producers[i]);
        }
    }
    for (int u : topo) {
        atom[u] = d.find(u);
        members[atom[u]].push_back(u);
        loads[atom[u]][g.ops[u].pipe] += g.ops[u].cycles;
        total[g.ops[u].pipe] += g.ops[u].cycles;
    }
    dominant = static_cast<int>(std::max_element(total.begin(), total.end()) - total.begin());
    dfs = dependency_dfs_order();
}
std::vector<int> CandidateGenerator::core_assignment(const Sequences &seq) const {
    std::vector<int> result(size, -1);
    for (int c = 0; c < static_cast<int>(seq.size()); ++c) {
        for (int u : seq[c]) {
            result[u] = c;
        }
    }
    return result;
}
Sequences CandidateGenerator::sequences_for_assignment(const std::vector<int> &assignment,
                                                       const std::vector<int> &order) const {
    Sequences seq(cores);
    for (int u : order) {
        seq[assignment[u]].push_back(u);
    }
    return seq;
}
double CandidateGenerator::resource_lower_bound(int active) const {
    double ddr = 0, forced = 0;
    for (const auto &t : problem.tensors) {
        if (t.producers.empty() && !t.consumers.empty()) {
            ddr += std::max(1.0, std::ceil(t.size / DDR_BANDWIDTH));
        }
        if (!t.producers.empty() && t.output) {
            ddr += std::max(1.0, std::ceil(t.size / DDR_BANDWIDTH));
        }
    }
    for (const auto &l : loads) {
        for (double v : l) {
            forced = std::max(forced, v);
        }
    }
    return std::max({ddr, forced, std::ceil(total[dominant] / active)});
}
std::vector<int> CandidateGenerator::dependency_dfs_order() const {
    std::vector<std::vector<int>> pred(size), succ(size), writers(problem.tensors.size());
    for (int u = 0; u < size; ++u) {
        for (int t : problem.ops[u].outputs) {
            writers[t].push_back(u);
        }
    }
    for (int u = 0; u < size; ++u) {
        for (int t : problem.ops[u].inputs) {
            pred[u].insert(pred[u].end(), writers[t].begin(), writers[t].end());
        }
        distinct(pred[u]);
        for (int p : pred[u]) {
            succ[p].push_back(u);
        }
    }
    std::vector<int> deg(size), d(size), queue;
    for (int u = 0; u < size; ++u) {
        deg[u] = static_cast<int>(pred[u].size());
        if (!deg[u]) {
            queue.push_back(u);
        }
    }
    for (std::size_t i = 0; i < queue.size(); ++i) {
        for (int v : succ[queue[i]]) {
            d[v] = std::max(d[v], d[queue[i]] + 1);
            if (!--deg[v]) {
                queue.push_back(v);
            }
        }
    }
    // 按 COPY 类别、依赖深度及 id 定义 DFS 栈顺序。
    auto less = [&](int a, int b, int copy) {
        return std::make_tuple(problem.ops[a].copy != copy, d[a], -problem.ops[a].id) <
               std::make_tuple(problem.ops[b].copy != copy, d[b], -problem.ops[b].id);
    };
    std::vector<int> stack, order;
    for (int u = 0; u < size; ++u) {
        if (succ[u].empty()) {
            stack.push_back(u);
        }
    }
    // 比较原图 DFS 的末端起点。
    std::sort(stack.begin(), stack.end(), [&](int a, int b) { return less(a, b, 2); });
    std::vector<char> seen(size);
    while (!stack.empty()) {
        int u = stack.back();
        if (seen[u]) {
            stack.pop_back();
            continue;
        }
        std::vector<int> parents;
        for (int p : pred[u]) {
            if (!seen[p]) {
                parents.push_back(p);
            }
        }
        if (parents.empty()) {
            seen[u] = true;
            stack.pop_back();
            if (!problem.ops[u].copy) {
                order.push_back(u);
            }
        } else {
            // 比较原图 DFS 的未访问前驱。
            std::sort(parents.begin(), parents.end(), [&](int a, int b) { return less(a, b, 1); });
            stack.insert(stack.end(), parents.begin(), parents.end());
        }
    }
    return order;
}
Grid CandidateGenerator::recognize_grid() const {
    Grid grid;
    for (int u : topo) {
        if (problem.ops[u].preds.size() > 1 || problem.ops[u].succs.size() > 1) {
            return grid;
        }
    }
    std::vector<std::vector<int>> chains, adjacency(problem.tensors.size());
    std::vector<std::pair<int, int>> input;
    for (int u : topo) {
        if (problem.ops[u].preds.empty()) {
            std::vector<int> chain{u};
            while (!problem.ops[chain.back()].succs.empty()) {
                chain.push_back(problem.ops[chain.back()].succs.front());
            }
            if (!chains.empty() && chain.size() != chains.front().size()) {
                return {};
            }
            std::vector<int> ext;
            for (int t : problem.ops[u].inputs) {
                if (problem.tensors[t].producers.empty()) {
                    ext.push_back(t);
                }
            }
            if (ext.size() != 2) {
                return {};
            }
            adjacency[ext[0]].push_back(ext[1]);
            adjacency[ext[1]].push_back(ext[0]);
            input.emplace_back(ext[0], ext[1]);
            chains.push_back(std::move(chain));
        }
    }
    if (chains.empty()) {
        return {};
    }
    std::vector<int> color(adjacency.size(), -1), row(adjacency.size(), -1), col(adjacency.size(), -1);
    for (int t = 0; t < static_cast<int>(adjacency.size()); ++t) {
        if (!adjacency[t].empty() && color[t] < 0) {
            color[t] = 0;
            std::vector<int> queue{t};
            for (std::size_t i = 0; i < queue.size(); ++i) {
                for (int v : adjacency[queue[i]]) {
                    if (color[v] < 0) {
                        color[v] = 1 - color[queue[i]];
                        queue.push_back(v);
                    } else if (color[v] == color[queue[i]]) {
                        return {};
                    }
                }
            }
        }
    }
    for (int t = 0; t < static_cast<int>(color.size()); ++t) {
        if (color[t] == 0) {
            row[t] = grid.rows++;
        }
        if (color[t] == 1) {
            col[t] = grid.cols++;
        }
    }
    if (static_cast<std::size_t>(grid.rows) * grid.cols != chains.size()) {
        return {};
    }
    grid.depth = static_cast<int>(chains.front().size());
    grid.cells.resize(chains.size());
    for (int i = 0; i < static_cast<int>(chains.size()); ++i) {
        int a = input[i].first, b = input[i].second;
        if (color[a]) {
            std::swap(a, b);
        }
        auto &target = grid.cells[row[a] * grid.cols + col[b]];
        if (!target.empty()) {
            return {};
        }
        target = std::move(chains[i]);
    }
    return grid;
}
std::vector<int> CandidateGenerator::coordinate_assignment(const std::vector<double> &coord, int active,
                                                           bool multi_pipe) const {
    std::vector<std::pair<double, int>> sorted;
    for (int a = 0; a < size; ++a) {
        if (!members[a].empty()) {
            double value = 0;
            for (int u : members[a]) {
                value += coord[u];
            }
            sorted.emplace_back(value / members[a].size(), a);
        }
    }
    std::sort(sorted.begin(), sorted.end());
    // 计算一个共享写者原子组的切分权重。
    auto weight = [&](int a) {
        double w = loads[a][dominant];
        if (multi_pipe) {
            w = 0;
            for (int p = 0; p < NUM_PIPES; ++p) {
                if (total[p]) {
                    w += loads[a][p] / total[p];
                }
            }
        }
        return w;
    };
    double sum = 0;
    for (auto entry : sorted) {
        sum += weight(entry.second);
    }
    std::vector<int> assignment(size, -1);
    double used = 0;
    for (auto entry : sorted) {
        int a = entry.second;
        double w = weight(a);
        int c = std::min(active - 1, static_cast<int>((used + w / 2) * active / std::max(1e-12, sum)));
        for (int u : members[a]) {
            assignment[u] = c;
        }
        used += w;
    }
    return assignment;
}
Sequences CandidateGenerator::delay_producers(Sequences seq, double factor, bool expanding_only,
                                              int selected_depth) const {
    const auto assignment = core_assignment(seq);
    std::vector<int> position(size);
    for (int c = 0; c < cores; ++c) {
        auto &order = seq[c];
        for (int i = 0; i < static_cast<int>(order.size()); ++i) {
            position[order[i]] = i;
        }
        const auto visit = order;
        for (auto it = visit.rbegin(); it != visit.rend(); ++it) {
            const int u = *it;
            if (selected_depth >= 0 && depth[u] != selected_depth) {
                continue;
            }
            const auto &op = problem.ops[u];
            if (op.succs.empty()) {
                continue;
            }
            int target = static_cast<int>(order.size());
            for (int v : op.succs) {
                if (assignment[v] != c) {
                    target = -1;
                    break;
                }
                target = std::min(target, position[v]);
            }
            --target;
            const int from = position[u];
            if (target <= from) {
                continue;
            }
            double inputs = 0, outputs = 0, cost = 0;
            for (int t : op.inputs) {
                inputs += problem.tensors[t].size;
            }
            for (int t : op.outputs) {
                outputs += problem.tensors[t].size;
            }
            if (expanding_only && inputs >= outputs) {
                continue;
            }
            for (int t : op.inputs) {
                if (std::find(op.outputs.begin(), op.outputs.end(), t) != op.outputs.end()) {
                    continue;
                }
                int last = -1;
                for (int v : problem.tensors[t].consumers) {
                    if (v != u && assignment[v] == c) {
                        last = std::max(last, position[v]);
                    }
                }
                for (int v : problem.tensors[t].producers) {
                    if (v != u && assignment[v] == c) {
                        last = std::max(last, position[v]);
                    }
                }
                if (last >= 0) {
                    cost += static_cast<double>(problem.tensors[t].size) *
                            std::max(0, target - std::max(from, last));
                }
            }
            if (!expanding_only && outputs * (target - from) <= factor * cost) {
                continue;
            }
            std::rotate(order.begin() + from, order.begin() + from + 1, order.begin() + target + 1);
            for (int i = from; i <= target; ++i) {
                position[order[i]] = i;
            }
        }
    }
    return seq;
}
CandidateGenerator::PlacementResources
CandidateGenerator::placement_resources(const std::vector<int> &assignment) const {
    std::array<std::array<double, NUM_PIPES>, 5> work{};
    for (int u : topo) {
        work[assignment[u]][problem.ops[u].pipe] += problem.ops[u].cycles;
    }
    double ddr = 0, peak = 0, sum = 0;
    for (const auto &core : work) {
        peak = std::max(peak, *std::max_element(core.begin(), core.end()));
        sum = std::max(sum, std::accumulate(core.begin(), core.end(), 0.0));
    }
    for (const auto &t : problem.tensors) {
        unsigned readers = 0;
        for (int u : t.consumers) {
            readers |= 1u << assignment[u];
        }
        int copies = 0;
        if (t.producers.empty()) {
            while (readers) {
                copies += readers & 1u;
                readers >>= 1;
            }
        } else {
            readers &= ~(1u << assignment[t.producers.front()]);
            while (readers) {
                copies += 2 * (readers & 1u);
                readers >>= 1;
            }
            copies += t.output;
        }
        ddr += copies * std::max(1.0, std::ceil(t.size / DDR_BANDWIDTH));
    }
    return {peak, ddr, sum};
}
double CandidateGenerator::placement_cost(const std::vector<int> &assignment) const {
    const auto r = placement_resources(assignment);
    return std::max(r.peak, r.ddr) + 0.15 * r.ddr + 0.05 * r.total_peak;
}
Sequences CandidateGenerator::window_order(const Sequences &reference, int window, double bias) const {
    const auto a = core_assignment(reference);
    std::vector<int> position(size), degree(size), cursor(cores);
    std::vector<char> done(size);
    for (int c = 0; c < cores; ++c) {
        for (int i = 0; i < static_cast<int>(reference[c].size()); ++i) {
            position[reference[c][i]] = i;
        }
    }
    // 容器元素类型：就绪窗口中的操作优先键。
    using Entry = std::pair<int, int>;
    std::vector<std::set<Entry>> ready(cores * NUM_PIPES);
    std::vector<std::vector<int>> uses(size);
    // 就绪窗口启发式中，一个张量在一个核心上的简化存活状态。
    struct Lifetime {
        int remaining = 0;
        bool resident = false;
    };
    std::vector<std::array<Lifetime, 5>> life(problem.tensors.size());
    for (int u : topo) {
        degree[u] = static_cast<int>(problem.ops[u].preds.size());
        if (!degree[u]) {
            ready[a[u] * NUM_PIPES + problem.ops[u].pipe].emplace(position[u], u);
        }
        uses[u] = problem.ops[u].inputs;
        uses[u].insert(uses[u].end(), problem.ops[u].outputs.begin(), problem.ops[u].outputs.end());
        distinct(uses[u]);
        for (int t : uses[u]) {
            ++life[t][a[u]].remaining;
        }
    }
    std::vector<std::array<double, NUM_PIPES>> pipes(cores);
    std::vector<double> finish(size), last(cores);
    Sequences result(cores);
    for (std::size_t placed = 0; placed < topo.size(); ++placed) {
        int chosen = -1;
        double chosen_start = 0;
        std::tuple<double, double, int, int> best_key;
        for (const auto &queue : ready) {
            int inspected = 0;
            for (auto entry : queue) {
                int u = entry.second, c = a[u];
                const auto &op = problem.ops[u];
                if (inspected++ == (window <= 64 ? 12 : 64) ||
                    (inspected > 1 && entry.first >= cursor[c] + window)) {
                    break;
                }
                double start = std::max(last[c], pipes[c][op.pipe]);
                for (int p : op.preds) {
                    start = std::max(start, finish[p] + (a[p] == c ? 0 : CROSS_CORE_DELAY));
                }
                std::array<double, 2> alloc{}, freed{};
                for (int t : uses[u]) {
                    int mem = problem.tensors[t].memory == 1 ? 0 : 1;
                    if (!life[t][c].resident) {
                        alloc[mem] += problem.tensors[t].size;
                    }
                    if (life[t][c].remaining == 1) {
                        freed[mem] += problem.tensors[t].size;
                    }
                }
                double delta = 0;
                for (int mem = 0; mem < 2; ++mem) {
                    double capacity = mem ? UB_CAPACITY : L1_CAPACITY;
                    delta += (alloc[mem] - freed[mem]) / capacity;
                }
                auto key = std::make_tuple(start + bias * (op.pipe != 0), delta, position[u] - cursor[c], u);
                if (chosen < 0 || key < best_key) {
                    chosen = u;
                    chosen_start = start;
                    best_key = key;
                }
            }
        }
        if (chosen < 0) {
            return reference;
        }
        const int u = chosen, c = a[u];
        const auto &op = problem.ops[u];
        ready[c * NUM_PIPES + op.pipe].erase({position[u], u});
        finish[u] = chosen_start + op.cycles;
        pipes[c][op.pipe] = finish[u];
        last[c] = chosen_start;
        for (int t : uses[u]) {
            auto &state = life[t][c];
            state.resident = --state.remaining != 0;
        }
        done[u] = true;
        result[c].push_back(u);
        while (cursor[c] < static_cast<int>(reference[c].size()) && done[reference[c][cursor[c]]]) {
            ++cursor[c];
        }
        for (int v : op.succs) {
            if (!--degree[v]) {
                ready[a[v] * NUM_PIPES + problem.ops[v].pipe].emplace(position[v], v);
            }
        }
    }
    return result;
}
Sequences CandidateGenerator::pull_consumers(Sequences seq, int selected_depth) const {
    const auto a = core_assignment(seq);
    std::vector<int> position(size);
    for (int c = 0; c < cores; ++c) {
        auto &order = seq[c];
        for (int i = 0; i < static_cast<int>(order.size()); ++i) {
            position[order[i]] = i;
        }
        const auto visit = order;
        for (int u : visit) {
            const auto &op = problem.ops[u];
            if (selected_depth >= 0 && depth[u] != selected_depth) {
                continue;
            }
            if (op.preds.empty()) {
                continue;
            }
            int target = 0;
            for (int p : op.preds) {
                if (a[p] == c) {
                    target = std::max(target, position[p] + 1);
                }
            }
            int from = position[u];
            if (target >= from) {
                continue;
            }
            std::rotate(order.begin() + target, order.begin() + from, order.begin() + from + 1);
            for (int i = target; i <= from; ++i) {
                position[order[i]] = i;
            }
        }
    }
    return seq;
}
}
}
