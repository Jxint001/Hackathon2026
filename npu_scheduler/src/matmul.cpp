#include "matmul.hpp"
#include "schedule.hpp"
#include "blocks.hpp"

#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace npu {
namespace {

// 只检查这批计算的共享关系，外围生产者可以位于区域之外。
bool match_grid(const Graph& g, const std::vector<std::vector<int>>& external,
                std::vector<std::vector<int>> chains, MatmulGrid& grid) {
    if (chains.empty()) return false;
    grid.depth = static_cast<int>(chains.front().size());
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
                for (int t : a) {
                    const auto& tensor = g.tensors[t];
                    grid.row_bytes = std::max(grid.row_bytes, tensor.bytes);
                    if (tensor.storage >= 0)
                        grid.row_pool[tensor.storage] = std::max(grid.row_pool[tensor.storage], tensor.bytes);
                }
                for (int t : b) {
                    const auto& tensor = g.tensors[t];
                    grid.column_bytes = std::max(grid.column_bytes, tensor.bytes);
                    if (tensor.storage >= 0)
                        grid.column_pool[tensor.storage] = std::max(grid.column_pool[tensor.storage], tensor.bytes);
                }
            }
        }
        if (!matched) return false;
    }
    // 统计累加输出的最大尺寸，供驻留容量估算使用。
    for (const auto& chain : grid.chains) for (int op : chain) {
        Cost bytes = 0;
        std::array<Cost, 2> pools{};
        for (int t : g.ops[op].outputs) {
            bytes += g.tensors[t].bytes;
            if (g.tensors[t].storage >= 0) pools[g.tensors[t].storage] += g.tensors[t].bytes;
        }
        for (int p = 0; p < 2; ++p) grid.partial_pool[p] = std::max(grid.partial_pool[p], pools[p]);
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
    CoreBlocks blocks(cores);
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
        for (const auto& tile : tiles) {
            ComputeBlock block; block.stages.resize(grid.depth);
            for (int k = 0; k < grid.depth; ++k)
                for (int cell : tile) block.stages[k].push_back(grid.chains[cell][k]);
            blocks[core].push_back(std::move(block));
        }
    }
    return emit_blocks(blocks, config.group, config.step);
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

std::vector<Partition> generate_core_partitions(const Graph& g, const MatmulGrid& grid, int cores, const GraphAnalysis& facts, const Plan& base) {
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
            Plan plan;
            if (!compose_local_plan(g, facts, base, build_tiled_plan(grid, candidate, cores, Configuration{0, 1, 1, 0, 1, grid.depth}), plan)) continue;
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

struct Panel {
    int r0, r1, c0, c1;
    bool reverse = false;
};

struct PanelLayout {
    bool transpose = false;
    std::vector<std::vector<Panel>> cores;
};

// 主体按完整面板分核，尾部沿另一维均衡分配。
PanelLayout make_panel_layout(const MatmulGrid& grid, int cores, int active, int height,
                              bool transpose, bool tail_first) {
    PanelLayout layout; layout.transpose = transpose; layout.cores.resize(cores);
    const int rows = transpose ? grid.columns : grid.rows;
    const int columns = transpose ? grid.rows : grid.columns;
    const int per_core = rows / (active * height) * height;
    const int tail = per_core * active;
    for (int c = 0; c < active; ++c) {
        auto& panels = layout.cores[c];
        for (int r = c * per_core; r < (c + 1) * per_core; r += height)
            panels.push_back(Panel{r, r + height, 0, columns, false});
        const int lo = (c * columns + active - 1) / active;
        const int hi = ((c + 1) * columns + active - 1) / active;
        if (tail < rows && lo < hi) {
            const Panel last{tail, rows, lo, hi, false};
            if (tail_first) panels.insert(panels.begin(), last); else panels.push_back(last);
        }
    }
    return layout;
}

Plan build_panel_plan(const MatmulGrid& grid, const PanelLayout& layout) {
    CoreBlocks blocks(layout.cores.size());
    for (int core = 0; core < static_cast<int>(layout.cores.size()); ++core) {
        int previous = -1;
        for (const auto& panel : layout.cores[core]) {
            const int width = panel.c1 - panel.c0;
            int first = panel.reverse ? panel.c1 - 1 : panel.c0;
            if (previous >= 0) first = std::max(panel.c0, std::min(panel.c1 - 1, previous));
            // 接续上一面板末列，允许反向和循环回绕。
            for (int j = 0; j < width; ++j) {
                const int offset = (first - panel.c0 + (panel.reverse ? -j : j) + width) % width;
                const int column = panel.c0 + offset;
                ComputeBlock block; block.stages.resize(grid.depth);
                for (int k = 0; k < grid.depth; ++k) for (int row = panel.r0; row < panel.r1; ++row) {
                    const int cell = layout.transpose ? column * grid.columns + row : row * grid.columns + column;
                    block.stages[k].push_back(grid.chains[cell][k]);
                }
                blocks[core].push_back(std::move(block));
                previous = column;
            }
        }
    }
    return emit_blocks(blocks, 1, grid.depth);
}

} // 匿名命名空间

std::vector<MatmulGrid> find_reuse_grids(const Graph& g, const GraphAnalysis& facts) {
    std::vector<std::vector<int>> operands(g.ops.size());
    std::vector<char> eligible(g.ops.size());
    for (int v : g.work) {
        for (int t : g.ops[v].inputs) if (facts.reader_count[t] > 1) operands[v].push_back(t);
        if (operands[v].size() != 2) {
            operands[v].clear();
            for (int t : g.ops[v].inputs) if (!facts.writer_count[t]) operands[v].push_back(t);
        }
        eligible[v] = operands[v].size() == 2;
    }
    // 核心段内保持单前驱、单后继；外围辅助操作不参与该条件。
    auto selected = eligible;
    for (int v : g.work) if (eligible[v]) {
        int predecessors = 0, successors = 0;
        for (int p : g.pred[v]) predecessors += eligible[p];
        for (int u : g.succ[v]) successors += eligible[u];
        if (predecessors > 1 || successors > 1) selected[v] = false;
    }
    std::vector<std::vector<int>> chains;
    for (int v : g.work) if (selected[v]) {
        bool root = true;
        for (int p : g.pred[v]) if (selected[p]) root = false;
        if (!root) continue;
        std::vector<int> chain;
        for (int u = v; u >= 0;) {
            chain.push_back(u);
            int next = -1;
            for (int w : g.succ[u]) if (selected[w]) next = w;
            u = next;
        }
        chains.push_back(std::move(chain));
    }
    UnionFind inputs(static_cast<int>(g.tensors.size()));
    for (const auto& chain : chains) inputs.join(operands[chain.front()][0], operands[chain.front()][1]);
    std::map<std::pair<int, int>, std::vector<std::vector<int>>> families;
    for (auto& chain : chains) {
        const auto key = std::make_pair(static_cast<int>(chain.size()), inputs.find(operands[chain.front()][0]));
        families[key].push_back(std::move(chain));
    }
    std::vector<MatmulGrid> result;
    for (auto& family : families) {
        MatmulGrid grid;
        if (!match_grid(g, operands, std::move(family.second), grid)) continue;
        if (grid.chains.size() < 4 && grid.op_count() != g.work.size()) continue;
        std::vector<char> member(g.ops.size());
        for (const auto& chain : grid.chains) for (int v : chain) member[v] = true;
        bool boundary = true;
        for (const auto& chain : grid.chains) for (int v : chain) for (int t : operands[v])
            for (int p : g.tensors[t].producers) if (member[p]) boundary = false;
        if (boundary) result.push_back(std::move(grid));
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.op_count() > b.op_count(); });
    return result;
}

namespace {

struct GridRecipe {
    Configuration config{};
    PanelLayout panels;
};

struct GridCatalog {
    const Graph& graph;
    const GraphAnalysis& facts;
    MatmulGrid grid;
    int cores;
    std::vector<Partition> partitions;
    std::map<PlanSearchState::PlanKey, GridRecipe> recipes;

    Plan finish(const Plan& local, const Plan& base, GridRecipe recipe) {
        Plan plan;
        if (!compose_local_plan(graph, facts, base, local, plan)) return {};
        recipes[fingerprint(plan)] = std::move(recipe);
        return plan;
    }
};

PlanAction tile_action(const std::shared_ptr<GridCatalog>& catalog, const std::shared_ptr<Plan>& base, Configuration config) {
    return {"grid-block", [catalog, base, config] {
        return catalog->finish(build_tiled_plan(catalog->grid, catalog->partitions[config.partition], catalog->cores, config), *base, GridRecipe{config, {}});
    }};
}

PlanAction panel_action(const std::shared_ptr<GridCatalog>& catalog, const std::shared_ptr<Plan>& base, PanelLayout layout) {
    return {"panel-block", [catalog, base, layout] {
        return catalog->finish(build_panel_plan(catalog->grid, layout), *base, GridRecipe{{}, layout});
    }};
}

std::vector<std::pair<int, int>> initial_shapes(const MatmulGrid& grid) {
    const auto resident_width = [&](bool rows) {
        int width = rows ? grid.rows : grid.columns;
        for (int p = 0; p < 2; ++p) {
            const Cost operand = rows ? grid.row_pool[p] : grid.column_pool[p];
            const Cost other = rows ? grid.column_pool[p] : grid.row_pool[p];
            const double available = static_cast<double>(kSpillThreshold[p]) - other - grid.partial_pool[p];
            const double per_chain = static_cast<double>(grid.depth) * operand + grid.partial_pool[p];
            if (per_chain > 0) width = std::min(width, static_cast<int>(std::max(0.0, available / per_chain)));
        }
        return std::max(1, width);
    };
    std::vector<std::pair<int, int>> shapes{{4, 4}, {resident_width(true), 1}, {1, resident_width(false)},
        {3, 1}, {1, 3}, {3, 6}, {6, 3}, {3, 3}, {2, 2}, {1, 1}, {3, 4}, {4, 3}};
    std::set<std::pair<int, int>> seen;
    shapes.erase(std::remove_if(shapes.begin(), shapes.end(), [&](auto& shape) {
        shape.first = std::min(shape.first, grid.rows); shape.second = std::min(shape.second, grid.columns);
        return !seen.insert(shape).second;
    }), shapes.end());
    return shapes;
}

std::vector<std::pair<int, int>> finer_shapes(const MatmulGrid& grid) {
    std::vector<std::pair<int, int>> shapes;
    for (int r = 1; r <= std::min(12, grid.rows); ++r) for (int c = 1; c <= std::min(12, grid.columns); ++c) {
        bool fits = true;
        for (int p = 0; p < 2; ++p) {
            const double bytes = static_cast<double>(r) * grid.row_pool[p] + static_cast<double>(c) * grid.column_pool[p] +
                (static_cast<double>(r) * c + 1) * grid.partial_pool[p];
            fits = fits && bytes <= 1.15 * kCapacity[p];
        }
        if (fits) shapes.emplace_back(r, c);
    }
    std::stable_sort(shapes.begin(), shapes.end(), [&](const auto& a, const auto& b) {
        const double x = (static_cast<double>(a.first) * grid.row_bytes + static_cast<double>(a.second) * grid.column_bytes) / (a.first * a.second);
        const double y = (static_cast<double>(b.first) * grid.row_bytes + static_cast<double>(b.second) * grid.column_bytes) / (b.first * b.second);
        return x != y ? x < y : a < b;
    });
    if (shapes.size() > 14) shapes.resize(14);
    return shapes;
}

// 从当前完整方案提取网格归属，允许通用动作的结果继续使用特化。
Partition grid_partition(const GridCatalog& catalog, const Plan& plan) {
    std::vector<int> owner(catalog.graph.ops.size(), -1);
    for (int c = 0; c < catalog.cores; ++c) for (int v : plan.orders[c]) owner[v] = c;
    Partition result(catalog.cores);
    const auto& grid = catalog.grid;
    for (int r = 0; r < grid.rows; ++r) for (int c = 0; c < grid.columns;) {
        const int core = owner[grid.chains[r * grid.columns + c][0]], start = c;
        while (c < grid.columns && owner[grid.chains[r * grid.columns + c][0]] == core) ++c;
        auto& rectangles = result[core];
        if (!rectangles.empty() && rectangles.back().r1 == r && rectangles.back().c0 == start && rectangles.back().c1 == c)
            rectangles.back().r1 = r + 1;
        else rectangles.push_back({r, r + 1, start, c});
    }
    return result;
}

} // 匿名命名空间

ActionProvider grid_actions(const Graph& graph, const GraphAnalysis& facts, const MatmulGrid& grid, int cores) {
    const auto catalog = std::make_shared<GridCatalog>(GridCatalog{graph, facts, grid, cores, {}, {}});
    ActionProvider provider; provider.coverage = grid.op_count();
    provider.initial = [catalog](const Plan& current) {
        const auto base = std::make_shared<Plan>(current);
        const auto& grid = catalog->grid; const int cores = catalog->cores;
        PlanActions actions;
        catalog->partitions = {Partition{{Rectangle{0, grid.rows, 0, grid.columns}}}};
        for (const auto& shape : std::vector<std::pair<int, int>>{{grid.rows, grid.columns}, {4, 4}, {3, 1}, {1, 3}})
            actions.push_back(tile_action(catalog, base, Configuration{0, shape.first, shape.second, 0, 1, grid.depth}));
        // 主体和尾部是块构造选项，与其他种子一起评估。
        for (int height : {3, 2, 4, 1, 6, 8}) for (bool transpose : {false, true}) for (bool tail_first : {true, false}) {
            if (height > (transpose ? grid.columns : grid.rows)) continue;
            actions.push_back(panel_action(catalog, base, make_panel_layout(grid, cores, cores, height, transpose, tail_first)));
        }
        // 独立输出计算段也可按工作量均衡，覆盖不规则负载。
        for (int active : {1, cores, std::max(1, cores - 1)}) for (bool dfs : {true, false})
            actions.push_back({"balance-output-blocks", [catalog, base, active, dfs] {
                const auto& graph = catalog->graph; const auto& chains = catalog->grid.chains;
                std::vector<Cost> weights(chains.size()), load(active);
                std::vector<int> sorted(chains.size()), owner(graph.ops.size(), -1), first(chains.size());
                std::iota(sorted.begin(), sorted.end(), 0);
                for (int i : sorted) {
                    first[i] = *std::min_element(chains[i].begin(), chains[i].end());
                    for (int v : chains[i]) weights[i] += std::max<Cost>(1, graph.ops[v].cycles);
                }
                std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                    return weights[a] != weights[b] ? weights[a] > weights[b] : first[a] < first[b];
                });
                for (int i : sorted) {
                    const int c = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
                    for (int v : chains[i]) owner[v] = c;
                    load[c] += weights[i];
                }
                Plan local, result; local.orders.resize(catalog->cores);
                for (int v : dependency_order(graph, dfs)) if (owner[v] >= 0) local.orders[owner[v]].push_back(v);
                if (!compose_local_plan(graph, catalog->facts, *base, local, result)) return Plan{};
                return result;
            }});
        const auto rectangles = generate_core_partitions(catalog->graph, grid, cores, catalog->facts, current);
        catalog->partitions.insert(catalog->partitions.end(), rectangles.begin(), rectangles.end());
        const auto shapes = initial_shapes(grid);
        for (int q = cores; q >= 2; --q) for (const auto& shape : shapes) {
            Partition round_robin(q); int index = 0;
            for (int r = 0; r < grid.rows; r += shape.first) for (int c = 0; c < grid.columns; c += shape.second)
                round_robin[index++ % q].push_back({r, std::min(r + shape.first, grid.rows), c, std::min(c + shape.second, grid.columns)});
            const int id = static_cast<int>(catalog->partitions.size()); catalog->partitions.push_back(std::move(round_robin));
            actions.push_back(tile_action(catalog, base, Configuration{id, shape.first, shape.second, 0, 1, grid.depth}));
        }
        for (const auto& shape : shapes) for (int p = 1; p <= static_cast<int>(rectangles.size()); ++p)
            actions.push_back(tile_action(catalog, base, Configuration{p, shape.first, shape.second, 0, 1, grid.depth}));
        return actions;
    };
    provider.neighbors = [catalog](const Candidate& seed, int round) {
        const auto base = std::make_shared<Plan>(seed.plan);
        const auto& grid = catalog->grid;
        PlanActions actions;
        auto found = catalog->recipes.find(fingerprint(seed.plan));
        GridRecipe recipe;
        if (found != catalog->recipes.end()) recipe = found->second;
        else {
            recipe.config = Configuration{static_cast<int>(catalog->partitions.size()), 3, 1, 0, 1, grid.depth};
            catalog->partitions.push_back(grid_partition(*catalog, seed.plan));
        }
        if (!recipe.panels.cores.empty()) {
            for (int c = 0; c < catalog->cores; ++c) {
                const int count = static_cast<int>(recipe.panels.cores[c].size());
                for (int p = 0; p < count; ++p) {
                    auto changed = recipe.panels; changed.cores[c][p].reverse = !changed.cores[c][p].reverse;
                    actions.push_back(panel_action(catalog, base, std::move(changed)));
                }
                for (int p = 1; p < count; ++p) {
                    auto changed = recipe.panels; std::swap(changed.cores[c][p - 1], changed.cores[c][p]);
                    actions.push_back(panel_action(catalog, base, std::move(changed)));
                }
            }
        } else {
            for (int group : {2, 4}) for (int step : {1, std::max(1, grid.depth / 2)}) for (int walk : {0, 1}) {
                auto config = recipe.config; config.group = group; config.step = step; config.walk = walk;
                actions.push_back(tile_action(catalog, base, config));
            }
            if (round < 2) for (const auto& shape : finer_shapes(grid)) for (int walk = 0; walk < 4; ++walk)
                actions.push_back(tile_action(catalog, base, Configuration{recipe.config.partition, shape.first, shape.second, walk, 1, grid.depth}));
        }
        return actions;
    };
    return provider;
}

} // npu 命名空间
