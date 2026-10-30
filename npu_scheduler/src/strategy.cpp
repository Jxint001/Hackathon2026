#include "strategy.hpp"
#include "model.hpp"
#include "conv.hpp"

#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace npu {
namespace {

struct MatmulGrid {
    int rows = 0, columns = 0, depth = 0; // 复用网格的行数、列数、累加链长度
    std::vector<std::vector<int>> chains; // 第 r 行 c 列的链存于 r * columns + c
    Cost row_bytes = 0, column_bytes = 0, partial_bytes = 0;
};

// 从依赖和共享输入识别规则网格，不依赖算子名或样例编号。
bool recognize_matmul_grid(const Graph& g, MatmulGrid& grid) {
    if (g.work.empty()) return false;
    std::vector<std::vector<int>> external(g.ops.size()), chains;
    // 没有计算写者的张量视为外部输入。
    std::vector<char> input(g.tensors.size(), 1);
    for (int t = 0; t < static_cast<int>(g.tensors.size()); ++t)
        for (int op : g.tensors[t].producers) if (!g.ops[op].copy) input[t] = 0;
    // 检查 PIPE_M、单链依赖，以及每个操作的两个外部输入。
    for (int op : g.work) {
        if (g.ops[op].pipe != 2 || g.pred[op].size() > 1 || g.succ[op].size() > 1) return false;
        for (int t : g.ops[op].inputs) if (input[t]) external[op].push_back(t);
        if (external[op].size() != 2) return false;
    }
    // 提取独立累加链：要求等长，并覆盖全部计算节点。
    std::size_t covered = 0;
    for (int root : g.work) if (g.pred[root].empty()) {
        std::vector<int> chain;
        for (int op = root; op >= 0; op = g.succ[op].empty() ? -1 : g.succ[op][0]) chain.push_back(op);
        covered += chain.size(); chains.push_back(std::move(chain));
    }
    if (covered != g.work.size() || chains.empty()) return false;
    grid.depth = static_cast<int>(chains[0].size());
    for (const auto& chain : chains) if (static_cast<int>(chain.size()) != grid.depth) return false;

    // 首层输入构成二部图，两侧分别对应网格的行、列。
    std::vector<std::vector<int>> adjacent(g.tensors.size());
    for (const auto& chain : chains) {
        const auto& in = external[chain[0]];
        adjacent[in[0]].push_back(in[1]); adjacent[in[1]].push_back(in[0]);
    }
    int start = 0;
    while (start < static_cast<int>(adjacent.size()) && adjacent[start].empty()) ++start;
    if (start == static_cast<int>(adjacent.size())) return false;
    std::vector<int> color(adjacent.size(), -1), queue{start}; color[start] = 0;
    for (std::size_t i = 0; i < queue.size(); ++i) for (int v : adjacent[queue[i]]) {
        if (color[v] == color[queue[i]]) return false;
        if (color[v] < 0) { color[v] = 1 - color[queue[i]]; queue.push_back(v); }
    }
    std::vector<int> row(adjacent.size(), -1), column(adjacent.size(), -1);
    for (int t = 0; t < static_cast<int>(adjacent.size()); ++t) if (!adjacent[t].empty()) {
        if (color[t] < 0) return false;
        if (color[t] == 0) row[t] = grid.rows++;
        else column[t] = grid.columns++;
    }
    // 每个行列组合必须恰好对应一条链。
    if (static_cast<std::size_t>(grid.rows) * grid.columns != chains.size()) return false;
    grid.chains.resize(chains.size());
    for (auto& chain : chains) {
        const auto& in = external[chain[0]];
        const int a = color[in[0]] == 0 ? in[0] : in[1];
        const int b = color[in[0]] == 1 ? in[0] : in[1];
        auto& destination = grid.chains[row[a] * grid.columns + column[b]];
        if (!destination.empty()) return false;
        destination = std::move(chain);
    }
    // 逐层验证：同行共享一个输入，同列共享另一个输入。
    for (int k = 0; k < grid.depth; ++k) {
        const auto& first = external[grid.chains[0][k]];
        bool matched = false;
        for (int orientation = 0; orientation < 2 && !matched; ++orientation) {
            std::vector<int> a(grid.rows, -1), b(grid.columns, -1);
            a[0] = first[orientation]; b[0] = first[1 - orientation];
            bool okay = true;
            for (int r = 0; r < grid.rows && okay; ++r) for (int c = 0; c < grid.columns; ++c) {
                const auto& in = external[grid.chains[r * grid.columns + c][k]];
                int x = in[0], y = in[1];
                if ((a[r] >= 0 && a[r] == y) || (b[c] >= 0 && b[c] == x)) std::swap(x, y);
                if ((a[r] >= 0 && a[r] != x) || (b[c] >= 0 && b[c] != y)) { okay = false; break; }
                a[r] = x; b[c] = y;
            }
            if (okay) {
                matched = true;
                for (int t : a) grid.row_bytes = std::max(grid.row_bytes, g.tensors[t].bytes);
                for (int t : b) grid.column_bytes = std::max(grid.column_bytes, g.tensors[t].bytes);
            }
        }
        if (!matched) return false;
    }
    // 统计累加输出的最大尺寸，供驻留容量估算使用。
    for (const auto& chain : grid.chains) for (int op : chain) {
        Cost bytes = 0;
        for (int t : g.ops[op].outputs) bytes += g.tensors[t].bytes;
        grid.partial_bytes = std::max(grid.partial_bytes, bytes);
    }
    return true;
}

struct Rectangle { int r0, r1, c0, c1; };
using Partition = std::vector<std::vector<Rectangle>>;
struct Configuration {
    int partition, rows, columns, walk = 0, group = 1, step = 1;
};

Plan build_tiled_plan(const MatmulGrid& grid, const Partition& partition, int cores,
               const Configuration& config) {
    Plan result; result.orders.resize(cores);
    for (int core = 0; core < static_cast<int>(partition.size()); ++core) {
        std::vector<std::vector<int>> tiles;
        for (const auto& region : partition[core]) {
            std::vector<std::vector<std::vector<int>>> stripes;
            const bool transpose = config.walk == 1 || config.walk == 3;
            const int outer_begin = transpose ? region.c0 : region.r0;
            const int outer_end = transpose ? region.c1 : region.r1;
            const int inner_begin = transpose ? region.r0 : region.c0;
            const int inner_end = transpose ? region.r1 : region.c1;
            const int outer_step = transpose ? config.columns : config.rows;
            const int inner_step = transpose ? config.rows : config.columns;
            for (int i = outer_begin; i < outer_end; i += outer_step) {
                std::vector<std::vector<int>> stripe;
                for (int j = inner_begin; j < inner_end; j += inner_step) {
                    std::vector<int> cells;
                    for (int a = i; a < std::min(i + outer_step, outer_end); ++a)
                        for (int b = j; b < std::min(j + inner_step, inner_end); ++b)
                            cells.push_back(transpose ? b * grid.columns + a : a * grid.columns + b);
                    stripe.push_back(std::move(cells));
                }
                if (config.walk >= 2 && stripes.size() % 2 == 1) std::reverse(stripe.begin(), stripe.end());
                stripes.push_back(std::move(stripe));
            }
            for (auto& stripe : stripes) for (auto& tile : stripe) tiles.push_back(std::move(tile));
        }
        auto& order = result.orders[core];
        for (int begin = 0; begin < static_cast<int>(tiles.size()); begin += config.group)
            for (int k = 0; k < grid.depth; k += config.step)
                for (int tile = begin; tile < std::min(begin + config.group, static_cast<int>(tiles.size())); ++tile)
                    for (int d = k; d < std::min(k + config.step, grid.depth); ++d)
                        for (int cell : tiles[tile]) order.push_back(grid.chains[cell][d]);
    }
    return result;
}

std::vector<std::vector<int>> compositions(int sum) {
    if (!sum) return {{}};
    std::vector<std::vector<int>> result;
    for (int first = 1; first <= sum; ++first)
        for (auto rest : compositions(sum - first)) {
            rest.insert(rest.begin(), first); result.push_back(std::move(rest));
        }
    return result;
}

std::vector<Partition> generate_core_partitions(const Graph& g, const MatmulGrid& grid, int cores) {
    std::vector<Partition> result;
    std::set<std::vector<int>> seen;
    for (int q = cores; q >= 1; --q) {
        struct Choice { Cost bound, ddr; Partition regions; std::vector<int> owner; bool band; };
        std::vector<Choice> choices;
        for (const auto& counts : compositions(q)) for (int transpose = 0; transpose < 2; ++transpose) {
            Partition candidate;
            std::vector<int> owner(grid.chains.size(), -1);
            const int outer = transpose ? grid.columns : grid.rows;
            const int inner = transpose ? grid.rows : grid.columns;
            int preceding = 0;
            for (int count : counts) {
                const int lo = preceding * outer / q, hi = (preceding + count) * outer / q;
                preceding += count;
                for (int index = 0; index < count; ++index) {
                    const int a = index * inner / count, b = (index + 1) * inner / count;
                    if (lo == hi || a == b) continue;
                    const Rectangle r = transpose ? Rectangle{a, b, lo, hi} : Rectangle{lo, hi, a, b};
                    const int core = static_cast<int>(candidate.size());
                    candidate.push_back({r});
                    for (int i = r.r0; i < r.r1; ++i) for (int j = r.c0; j < r.c1; ++j)
                        owner[i * grid.columns + j] = core;
                }
            }
            if (candidate.empty() || !seen.insert(owner).second) continue;
            const auto plan = build_tiled_plan(grid, candidate, cores, Configuration{0, 1, 1, 0, 1, grid.depth});
            choices.push_back(Choice{partition_lower_bound(g, plan), partition_ddr_cycles(g, plan), std::move(candidate), std::move(owner),
                                     counts.size() == 1 || counts.size() == static_cast<std::size_t>(q)});
        }
        std::stable_sort(choices.begin(), choices.end(), [](const Choice& a, const Choice& b) {
            return std::tie(a.bound, a.ddr, a.owner) < std::tie(b.bound, b.ddr, b.owner);
        });
        int selected = 0;
        for (auto& choice : choices) if (selected < 4 || choice.band) {
            result.push_back(std::move(choice.regions)); ++selected;
        }
    }
    return result;
}

Cost mandatory_bytes(const Graph& g) {
    Cost result = 0;
    for (const auto& t : g.tensors) {
        bool producer = false, consumer = false, output = false;
        for (int op : t.producers) producer = producer || !g.ops[op].copy;
        for (int op : t.consumers) {
            consumer = consumer || !g.ops[op].copy; output = output || g.ops[op].copy == 2;
        }
        if ((!producer && consumer) || (producer && (output || !consumer))) {
            if (result > kInfinity - t.bytes) return kInfinity;
            result += t.bytes;
        }
    }
    return result;
}

// 保存评估次数、已试方案和最优结果；构造时只初始化这些记录。
struct PlanSearchState {
    const Graph& graph;
    Cost time_lower_bound, minimum_copy_bytes;
    Plan best_plan;
    Evaluation best_score;
    std::set<std::pair<std::uint64_t, std::uint64_t>> tried_plan_hashes;
    int evaluated_count = 0, max_evaluations;
    explicit PlanSearchState(const Graph& g, int cores) : graph(g), time_lower_bound(universal_lower_bound(g, cores)),
        minimum_copy_bytes(mandatory_bytes(g)),
        max_evaluations(static_cast<int>(std::min<std::size_t>(640, std::max<std::size_t>(12, 9000000 / std::max<std::size_t>(1, g.work.size()))))) {}

    // 次数用完，或时间和搬运量都达到下界时停止。
    bool should_stop() const {
        return evaluated_count >= max_evaluations ||
               (best_score.valid && best_score.makespan == time_lower_bound && best_score.bytes == minimum_copy_bytes);
    }
    // 去重、剪枝、预测性能，并更新当前最优方案。
    Evaluation evaluate_candidate(const Plan& plan) {
        Evaluation result;
        if (should_stop()) return result;
        std::uint64_t a = 1469598103934665603ULL, b = 0x9e3779b97f4a7c15ULL;
        for (const auto& core : plan.orders) {
            a = (a ^ UINT64_MAX) * 1099511628211ULL;
            b = b * 6364136223846793005ULL + core.size();
            for (int op : core) { a = (a ^ static_cast<std::uint64_t>(op)) * 1099511628211ULL; b = (b ^ a) * 2862933555777941757ULL + 3037000493ULL; }
        }
        if (!tried_plan_hashes.emplace(a, b).second) return result;
        if (partition_lower_bound(graph, plan) > best_score.makespan) return result;
        ++evaluated_count;
        result = evaluate(graph, plan, best_score.makespan);
        if (result.valid && (!best_score.valid ||
            std::tie(result.makespan, result.bytes, plan.orders) < std::tie(best_score.makespan, best_score.bytes, best_plan.orders))) {
            best_score = result; best_plan = plan;
            // 已接近时间下界时，缩减后续评估次数。
            if (static_cast<double>(best_score.makespan - time_lower_bound) <= 0.005 * static_cast<double>(best_score.makespan))
                max_evaluations = std::min(max_evaluations, std::max(24, evaluated_count + 8));
        }
        return result;
    }
};

// 按独立依赖分量均衡分核，补充 Matmul 的候选方案。
void try_load_balanced_plans(const Graph& g, int cores, PlanSearchState& state) {
    std::vector<int> component(g.ops.size(), -1);
    std::vector<std::vector<int>> groups;
    for (int root : g.work) if (component[root] < 0) {
        const int id = static_cast<int>(groups.size());
        std::vector<int> queue{root}; component[root] = id;
        for (std::size_t i = 0; i < queue.size(); ++i) {
            const int op = queue[i];
            for (const auto* list : {&g.pred[op], &g.succ[op]}) for (int v : *list)
                if (component[v] < 0) { component[v] = id; queue.push_back(v); }
        }
        groups.push_back(std::move(queue));
    }
    std::vector<Cost> weights(groups.size());
    std::vector<int> sorted(groups.size()); std::iota(sorted.begin(), sorted.end(), 0);
    for (int i : sorted) for (int op : groups[i]) weights[i] += std::max<Cost>(1, g.ops[op].cycles);
    std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return weights[a] > weights[b]; });
    const auto dfs = dependency_order(g, true);
    for (int q : {1, cores, std::max(1, cores - 1)}) {
        std::vector<Cost> load(q);
        std::vector<int> owner(groups.size());
        for (int group : sorted) {
            const int core = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
            owner[group] = core; load[core] += weights[group];
        }
        for (const auto* order : {&dfs, &g.topo}) {
            Plan plan; plan.orders.resize(cores);
            for (int op : *order) plan.orders[owner[component[op]]].push_back(op);
            state.evaluate_candidate(plan);
        }
    }
}

// 尝试不同分核、块大小和执行顺序，评估结果记入 state。
void try_tiled_matmul_plans(const Graph& g, const MatmulGrid& grid, int cores, PlanSearchState& state) {
    // 先试单核，搬运占主导时多核未必更快。
    const Partition single{{Rectangle{0, grid.rows, 0, grid.columns}}};
    for (const auto& shape : std::vector<std::pair<int, int>>{{grid.rows, grid.columns}, {4, 4}, {3, 1}, {1, 3}})
        state.evaluate_candidate(build_tiled_plan(grid, single, cores, Configuration{0, shape.first, shape.second, 0, 1, grid.depth}));
    if (state.should_stop()) return;
    // 生成按行、列和二维矩形划分的候选。
    auto regions = generate_core_partitions(g, grid, cores);

    struct Seed { Evaluation score; Configuration config; };
    std::vector<Seed> seeds;
    // 将参数展开为方案，评估后保存有效候选。
    const auto test = [&](Configuration config) {
        const Plan plan = build_tiled_plan(grid, regions[config.partition], cores, config);
        const Evaluation score = state.evaluate_candidate(plan);
        if (score.valid) seeds.push_back(Seed{score, config});
    };
    // 按 90% L1 估算能保留整段累加输入的细长块宽度。
    const auto resident_width = [&](Cost operand, Cost other, int dimension) {
        const double available = static_cast<double>(kSpillThreshold[0]) - other - grid.partial_bytes;
        const double per_chain = static_cast<double>(grid.depth) * operand + grid.partial_bytes;
        if (per_chain <= 0) return dimension;
        return std::max(1, std::min(dimension, static_cast<int>(std::max(0.0, available / per_chain))));
    };
    const int resident_rows = resident_width(grid.row_bytes, grid.column_bytes, grid.rows);
    const int resident_columns = resident_width(grid.column_bytes, grid.row_bytes, grid.columns);
    // 常见块形状，加上按容量估算的细长块。
    std::vector<std::pair<int, int>> initial{{4, 4}, {resident_rows, 1}, {1, resident_columns},
        {3, 1}, {1, 3}, {3, 6}, {6, 3}, {3, 3}, {2, 2}, {1, 1}, {3, 4}, {4, 3}};
    std::set<std::pair<int, int>> unique_shapes;
    initial.erase(std::remove_if(initial.begin(), initial.end(), [&](auto& shape) {
        shape.first = std::min(shape.first, grid.rows); shape.second = std::min(shape.second, grid.columns);
        return !unique_shapes.insert(shape).second;
    }), initial.end());
    const int rectangular = static_cast<int>(regions.size());
    // 小块轮流分给各核，补充负载均衡候选。
    for (int q = cores; q >= 2 && !state.should_stop(); --q) for (const auto& shape : initial) {
        if (state.should_stop()) break;
        Partition rr(q);
        int index = 0;
        for (int r = 0; r < grid.rows; r += shape.first) for (int c = 0; c < grid.columns; c += shape.second) {
            rr[index++ % q].push_back(Rectangle{r, std::min(r + shape.first, grid.rows), c, std::min(c + shape.second, grid.columns)});
        }
        const int id = static_cast<int>(regions.size()); regions.push_back(std::move(rr));
        test(Configuration{id, shape.first, shape.second, 0, 1, grid.depth});
    }
    // 同样的块形状也用于矩形分核候选。
    for (const auto& shape : initial) for (int p = 0; p < rectangular && !state.should_stop(); ++p)
        test(Configuration{p, shape.first, shape.second, 0, 1, grid.depth});
    if (state.should_stop() || seeds.empty()) return;
    // 按时间、搬运量排序，保留至多 4 种不同分核。
    std::stable_sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        return std::tie(a.score.makespan, a.score.bytes) < std::tie(b.score.makespan, b.score.bytes);
    });
    std::vector<Configuration> beam;
    std::set<int> selected;
    for (const auto& seed : seeds) if (selected.insert(seed.config.partition).second) {
        beam.push_back(seed.config); if (beam.size() == 4) break;
    }
    // 调整交错块数、累加步长和行列遍历。
    for (int group : {2, 4}) for (int step : {1, std::max(1, grid.depth / 2)})
        for (auto config : beam) for (int walk : {0, 1}) {
            config.group = group; config.step = step; config.walk = walk;
            test(config);
        }
    // 容量粗估筛选更多块形状，最终优劣由完整性能模型决定。
    std::vector<std::pair<int, int>> shapes;
    for (int r = 1; r <= std::min(12, grid.rows); ++r)
        for (int c = 1; c <= std::min(12, grid.columns); ++c) {
            const double bytes = static_cast<double>(r) * grid.row_bytes + static_cast<double>(c) * grid.column_bytes +
                                 (static_cast<double>(r) * c + 1) * grid.partial_bytes;
            if (bytes <= 1.15 * kCapacity[0]) shapes.emplace_back(r, c);
        }
    // 按每条链的平均输入量排序，最多保留 14 种形状。
    std::stable_sort(shapes.begin(), shapes.end(), [&](const auto& a, const auto& b) {
        const double x = (static_cast<double>(a.first) * grid.row_bytes + static_cast<double>(a.second) * grid.column_bytes) / (a.first * a.second);
        const double y = (static_cast<double>(b.first) * grid.row_bytes + static_cast<double>(b.second) * grid.column_bytes) / (b.first * b.second);
        return x != y ? x < y : a < b;
    });
    if (shapes.size() > 14) shapes.resize(14);
    // 比较行、列和两种蛇形遍历。
    for (const auto& shape : shapes) for (const auto& seed : beam) for (int walk = 0; walk < 4 && !state.should_stop(); ++walk)
        test(Configuration{seed.partition, shape.first, shape.second, walk, 1, grid.depth});
    if (state.should_stop()) return;
    // 从全部有效候选取前 4 名，再细调交错块数和累加步长。
    std::stable_sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        return std::tie(a.score.makespan, a.score.bytes) < std::tie(b.score.makespan, b.score.bytes);
    });
    std::vector<Configuration> best;
    for (std::size_t i = 0; i < std::min<std::size_t>(4, seeds.size()); ++i) best.push_back(seeds[i].config);
    for (int group : {2, 4}) for (int step : {1, std::max(1, grid.depth / 2)}) for (auto config : best) {
        config.group = group; config.step = step;
        test(config);
    }
}

Plan solve_matmul(const Graph& graph, const MatmulGrid& grid, int cores) {
    // 1. 初始化搜索次数上限和最优结果记录。
    PlanSearchState state(graph, cores);

    // 2. 尝试不同分块、分核和核内执行顺序。
    try_tiled_matmul_plans(graph, grid, cores, state);

    // 3. 仍需搜索时，再尝试按计算量均衡分核。
    if (!state.should_stop()) {
        try_load_balanced_plans(graph, cores, state);
    }

    // 4. 返回所有已试方案中最好的一个。
    return state.best_plan;
}

} // 匿名命名空间

Plan solve(const Graph& graph, int cores) {
    MatmulGrid grid;
    if (recognize_matmul_grid(graph, grid)) {
        // 规则 Matmul：网格分块搜索，再补充独立分量分核。
        return solve_matmul(graph, grid, cores);
    } else {
        // Conv 等非规则图：区域分组与联合分核、排序搜索。
        return solve_general(graph, cores);
    }
}
} // namespace npu
