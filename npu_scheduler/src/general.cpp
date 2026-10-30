#include "general.hpp"
#include "search.hpp"
#include "analysis.hpp"
#include "schedule.hpp"

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

using Changes = std::vector<std::pair<int, int>>;

double band_step(const SchedulingGroups& r, int cores) {
    std::map<int, std::vector<double>> layers;
    for (int v : r.topo) if (r.units[v].coordinate >= 0) layers[r.units[v].depth].push_back(r.units[v].coordinate);
    double step = 1;
    for (auto& item : layers) {
        auto& positions = item.second; unique_sort(positions);
        if (positions.size() < static_cast<std::size_t>(2 * cores)) continue;
        std::vector<double> gaps;
        for (std::size_t i = 1; i < positions.size(); ++i) gaps.push_back(positions[i] - positions[i - 1]);
        const double spacing = median_position(std::move(gaps));
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
    const SchedulingGroups& r;
    std::vector<int> owner;
    std::vector<std::array<int, 5>> producers, consumers;
    std::vector<std::array<Cost, kPipeCount>> load;
    Cost ddr = 0;
    Placement(const SchedulingGroups& regions, std::vector<int> assignment, int cores)
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

// 只生成核归属，按粗估筛选后交给排序函数。
std::vector<std::vector<int>> make_initial_placements(const SchedulingGroups& r, int cores) {
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
        owner = r.normalize_owners(std::move(owner));
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
Plan delay_expanding_producers(const SchedulingGroups& r, Plan p) {
    p = delay_grouped_producers(r, p, 1, true, false);
    return delay_grouped_producers(r, p, 1, true, true);
}

struct Move { Changes changes; double estimate; int kind; };

// 生成迁移、拆分、交换和条带调整候选，先做静态筛选。
std::vector<Move> propose_moves(const SchedulingGroups& r, const Candidate& seed, int cores, Cost incumbent) {
    Placement state(r, r.owners(seed.plan), cores);
    const auto owner = state.owner;
    const double baseline = state.estimate();
    std::array<std::vector<Move>, 5> pools;
    const auto offer = [&](Changes changes, int kind) {
        changes = r.close_changes(changes);
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
            step = gaps.empty() ? 1.0 : median_position(std::move(gaps));
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

} // 匿名命名空间

ActionProvider general_actions(std::shared_ptr<const SchedulingGroups> groups, int cores, std::size_t coverage) {
    ActionProvider provider; provider.coverage = coverage;
    provider.initial = [groups, cores](const Plan&) {
        const auto& r = *groups;
        PlanActions actions;
        std::vector<int> single(r.units.size());
        actions.push_back({"group-reference", [groups, cores, single] { return groups->project(single, groups->dfs, cores); }});
        actions.push_back({"group-topo", [groups, cores, single] { return groups->project(single, groups->topo, cores); }});
        std::vector<Cost> durations;
        for (const auto& unit : r.units) if (unit.work[2] > 1) durations.push_back(unit.work[2]);
        std::sort(durations.begin(), durations.end());
        const double bias = durations.empty() ? 0 : (r.compact ? 2.0 : 2.5) * durations[durations.size() / 2];
        for (const auto& owner : make_initial_placements(r, cores)) {
            actions.push_back({"placement-reference", [groups, cores, owner] { return groups->project(owner, groups->dfs, cores); }});
            for (double preference : {0.0, bias}) for (bool delay : {false, true}) {
                actions.push_back({"placement-order", [groups, cores, owner, preference, delay] {
                    const auto& r = *groups;
                    const auto order = make_ready_order(r, owner, cores, r.dfs, ReadyPolicy{4096, r.compact ? 12 : 64, 1, preference, 0});
                    auto plan = r.project(owner, order, cores);
                    return delay ? delay_expanding_producers(r, std::move(plan)) : plan;
                }});
            }
        }
        return actions;
    };
    provider.neighbors = [groups, cores](const Candidate& seed, int round) {
        const auto& r = *groups;
        PlanActions actions;
        const auto base = std::make_shared<Plan>(seed.plan);
        const auto owner = r.owners(seed.plan);
        const auto reference = reference_group_order(r, seed.plan, seed.score);
        for (double factor : {1.0, 0.5, 2.0})
            actions.push_back({"delay-block", [groups, base, factor] { return delay_grouped_producers(*groups, *base, factor, false, false); }});
        std::map<int, Cost> pressure;
        if (!r.compact) for (int v : r.topo) for (int t : r.units[v].outputs) pressure[r.units[v].depth] += seed.score.tensor_spill_bytes[t];
        std::vector<std::pair<Cost, int>> stages;
        for (const auto& item : pressure) if (item.second) stages.emplace_back(-item.second, item.first);
        std::sort(stages.begin(), stages.end());
        if (stages.size() > 3) stages.resize(3);
        for (const auto& stage : stages) {
            const int depth = stage.second;
            actions.push_back({"delay-spill-stage", [groups, base, depth] { return delay_grouped_producers(*groups, *base, .5, false, false, depth); }});
        }
        for (int window : {64, 512}) actions.push_back({"reorder-blocks", [groups, cores, owner, reference, window] {
            const auto& r = *groups;
            const auto order = make_ready_order(r, owner, cores, reference, ReadyPolicy{window, r.compact ? 12 : 64, 1, 0, 4.0 * kCrossCoreDelay});
            return delay_grouped_producers(r, r.project(owner, order, cores), 1, false, false);
        }});
        if (round > 0) {
            // 真正取到迁移动作时才计算边界候选，避免提前展开未使用的动作。
            const auto moves = std::make_shared<std::vector<Move>>();
            const auto prepared = std::make_shared<bool>(false);
            const auto score = std::make_shared<Evaluation>(seed.score);
            for (int index = 0; index < 15; ++index) for (bool reorder : {false, true})
                actions.push_back({"move-block", [groups, cores, base, owner, reference, score, moves, prepared, index, reorder] {
                const auto& r = *groups;
                if (!*prepared) {
                    *moves = propose_moves(r, Candidate{*base, *score}, cores, score->makespan + score->makespan / 20);
                    *prepared = true;
                }
                if (index >= static_cast<int>(moves->size())) return Plan{};
                auto assignment = owner;
                for (const auto& change : (*moves)[index].changes) assignment[change.first] = change.second;
                if (!reorder) return delay_grouped_producers(r, r.project(assignment, reference, cores), 1, false, false);
                const auto order = make_ready_order(r, assignment, cores, reference, ReadyPolicy{512, r.compact ? 12 : 64});
                return delay_expanding_producers(r, r.project(assignment, order, cores));
            }});
        }
        return actions;
    };
    return provider;
}

} // npu 命名空间
