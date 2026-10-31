#include "thread_pool.hpp"
#include "search/common.hpp"
#include "search/search_controller.hpp"
namespace npu {
namespace search_detail {
void SearchController::grid_search(const Grid &grid) {
    const std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3}, {3, 4}, {4, 3}, {4, 4},
        {4, 5}, {5, 4}, {2, 6}, {6, 2}, {1, 3}, {3, 1}, {1, 4}, {4, 1}, {1, 5}, {5, 1},
        {1, 6}, {6, 1}, {2, 4}, {4, 2}, {2, 5}, {5, 2}, {3, 5}, {5, 3}, {3, 6}, {6, 3}};
    for (int active = cores; active >= 1; --active) {
        if (generator.resource_lower_bound(active) > state.best.score.makespan) {
            continue;
        }
        std::vector<Candidate> batch;
        for (int axis = 0; axis < 2; ++axis) {
            for (auto shape : shapes) {
                for (int placement = 0; placement < 3; ++placement) {
                    // 容器键类型：网格候选中的分块分组键。
                    using Tile = std::tuple<int, int, int>;
                    std::map<Tile, std::vector<int>> tiles;
                    int count = grid.rows * grid.cols;
                    for (int i = 0; i < count; ++i) {
                        int r = axis ? i % grid.rows : i / grid.cols;
                        int c = axis ? i / grid.rows : i % grid.cols;
                        int core = placement == 0 ? std::min(active - 1, i * active / count) : 0;
                        int a = r / shape.first, b = c / shape.second;
                        tiles[{core, axis ? b : a, axis ? a : b}].push_back(r * grid.cols + c);
                    }
                    Sequences seq(cores);
                    std::vector<int> load(active);
                    int tile_index = 0;
                    for (const auto &tile : tiles) {
                        int core = std::get<0>(tile.first);
                        if (placement == 1) {
                            core = tile_index % active;
                        }
                        if (placement == 2) {
                            core =
                                static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
                        }
                        for (int k = 0; k < grid.depth; ++k) {
                            for (int cell : tile.second) {
                                seq[core].push_back(grid.cells[cell][k]);
                            }
                        }
                        load[core] += static_cast<int>(tile.second.size());
                        ++tile_index;
                    }
                    batch.push_back({std::move(seq)});
                    if (batch.size() == 10) {
                        consider_batch(batch);
                        batch.clear();
                    }
                }
            }
        }
        consider_batch(batch);
    }
}

void SearchController::granularity() {
    auto trace = trace_best();
    std::vector<Candidate> batch;
    for (int chunk : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}) {
        batch.push_back({trace.sequences, chunk});
    }
    consider_batch(batch);
}

void SearchController::aligned_tiles() {
    const auto grid = generator.recognize_grid();
    if (!grid.rows) {
        return;
    }
    for (int u : generator.topo) {
        if (problem.ops[u].pipe != 0) {
            return;
        }
    }
    // 一条由多个小块组成的矩阵条带。
    using Stripe = std::vector<std::vector<int>>;
    // 某个核心按执行先后排列的多个矩阵条带。
    using Stripes = std::vector<Stripe>;
    // 对齐条带搜索中的完整多核布局，外层下标为核心编号。
    using Layout = std::vector<Stripes>;
    // 将条带和块布局展开成各核逐层计算序列。
    auto sequence = [&](const Layout &layout) {
        Sequences seq(cores);
        for (int c = 0; c < cores; ++c) {
            for (const auto &stripe : layout[c]) {
                for (const auto &tile : stripe) {
                    for (int d = 0; d < grid.depth; ++d) {
                        for (int cell : tile) {
                            seq[c].push_back(grid.cells[cell][d]);
                        }
                    }
                }
            }
        }
        return seq;
    };
    Layout chosen;
    double chosen_time = std::numeric_limits<double>::infinity();
    std::vector<Layout> pending;
    // 复核待处理条带布局并更新局部和全局最优。
    auto flush = [&](bool limited) {
        std::vector<Candidate> batch;
        for (const auto &layout : pending) {
            batch.push_back({sequence(layout)});
        }
        prefetch(batch);
        for (std::size_t i = 0; i < pending.size(); ++i) {
            auto r = consider(batch[i].seq);
            if (r.valid && r.makespan < chosen_time - 0.5) {
                chosen_time = r.makespan;
                chosen = pending[i];
            }
            if (limited && state.phase_attempts >= std::min(state.phase_limit, 192)) {
                pending.clear();
                return true;
            }
        }
        pending.clear();
        return false;
    };
    // 暂存条带布局，凑满十个后批量复核。
    auto test = [&](const Layout &layout, bool limited = false) {
        pending.push_back(layout);
        return pending.size() == 10 && flush(limited);
    };

    for (int active = cores; active >= 2; --active) {
        for (int transpose : {0, 1}) {
            if (generator.resource_lower_bound(active) > state.best.score.makespan) {
                continue;
            }
            int outer = transpose ? grid.cols : grid.rows, inner = transpose ? grid.rows : grid.cols;
            for (int height : {1, 2, 3, 4}) {
                int main = outer / (active * height) * height;
                if (!main) {
                    continue;
                }
                Layout layout(cores);
                for (int c = 0; c < active; ++c) {
                    // 把矩形区域按条带高度加入当前核心布局。
                    auto append = [&](int lo, int hi, int begin, int end) {
                        for (int r = lo; r < hi; r += height) {
                            Stripe stripe;
                            for (int col = begin; col < end; ++col) {
                                std::vector<int> tile;
                                for (int row = r; row < std::min(r + height, hi); ++row) {
                                    tile.push_back(transpose ? col * grid.cols + row : row * grid.cols + col);
                                }
                                stripe.push_back(std::move(tile));
                            }
                            if (!stripe.empty()) {
                                layout[c].push_back(std::move(stripe));
                            }
                        }
                    };
                    append(c * main, (c + 1) * main, 0, inner);
                    append(active * main, outer, (c * inner + active - 1) / active,
                           ((c + 1) * inner + active - 1) / active);
                }
                test(layout);
                for (auto &stripes : layout) {
                    for (int i = 1; i < static_cast<int>(stripes.size()); i += 2) {
                        std::reverse(stripes[i].begin(), stripes[i].end());
                    }
                }
                test(layout);
            }
            flush(false);
        }
    }
    if (chosen.empty()) {
        return;
    }

    for (int round = 0; round < 2; ++round) {
        const double before = chosen_time;
        for (int c = 0; c < cores; ++c) {
            const auto seed = chosen;
            int n = static_cast<int>(seed[c].size());
            if (!n) {
                continue;
            }
            for (int start = 0; start < std::min(n, 4); ++start) {
                for (int mode = 0; mode < 4; ++mode) {
                    auto trial_layout = seed;
                    auto &stripes = trial_layout[c];
                    std::rotate(stripes.begin(), stripes.begin() + start, stripes.end());
                    for (int i = 0; i < n; ++i) {
                        auto &stripe = stripes[i];
                        if ((mode & 1) || ((mode & 2) && (i & 1))) {
                            std::reverse(stripe.begin(), stripe.end());
                        }
                        if (!i) {
                            continue;
                        }

                        const auto &last = stripes[i - 1].back();
                        int match = -1, overlap = 0;
                        for (int j = 0; j < static_cast<int>(stripe.size()); ++j) {
                            int score = 0;
                            for (int a : last) {
                                for (int b : stripe[j]) {
                                    score +=
                                        (a / grid.cols == b / grid.cols) + (a % grid.cols == b % grid.cols);
                                }
                            }
                            if (score > overlap) {
                                overlap = score;
                                match = j;
                            }
                        }
                        if (match > 0) {
                            std::rotate(stripe.begin(), stripe.begin() + match, stripe.end());
                        }
                    }
                    if (test(trial_layout, true)) {
                        return;
                    }
                }
            }
            if (flush(true)) {
                return;
            }
        }
        if (chosen_time >= before - 0.5) {
            break;
        }
    }
}

void SearchController::capacity_tiles() {
    const auto grid = generator.recognize_grid();
    if (!grid.rows) {
        return;
    }
    for (int u : generator.topo) {
        if (problem.ops[u].pipe != 0) {
            return;
        }
    }
    double operand = 0, partial = 0;
    for (int u : generator.topo) {
        double output_bytes = 0;
        for (int t : problem.ops[u].inputs) {
            if (problem.tensors[t].producers.empty()) {
                operand = std::max(operand, static_cast<double>(problem.tensors[t].size));
            }
        }
        for (int t : problem.ops[u].outputs) {
            output_bytes += problem.tensors[t].size;
        }
        partial = std::max(partial, output_bytes);
    }
    // 容量分块搜索中分给一个核心的网格矩形区域。
    struct Rect {
        int r0, r1, c0, c1;
    };
    // 容量分块搜索的一份多核矩形分区。
    using RectanglesByCore = std::vector<std::vector<Rect>>;
    std::vector<std::pair<double, RectanglesByCore>> partitions;
    std::set<std::vector<int>> placements;
    for (int active = cores; active >= 1; --active) {
        if (generator.resource_lower_bound(active) > state.best.score.makespan) {
            continue;
        }
        for (int mask = 0; mask < (1 << (active - 1)); ++mask) {
            for (int transpose = 0; transpose < 2; ++transpose) {
                std::vector<int> counts;
                int length = 1;
                for (int i = 0; i < active - 1; ++i) {
                    if (mask & (1 << i)) {
                        counts.push_back(length);
                        length = 1;
                    } else {
                        ++length;
                    }
                }
                counts.push_back(length);
                RectanglesByCore regions(cores);
                std::vector<int> assignment(size, -1);
                int preceding = 0, c = 0;
                int outer = transpose ? grid.cols : grid.rows, inner = transpose ? grid.rows : grid.cols;
                for (int count : counts) {
                    int lo = preceding * outer / active, hi = (preceding + count) * outer / active;
                    preceding += count;
                    for (int i = 0; i < count; ++i, ++c) {
                        int begin = i * inner / count, end = (i + 1) * inner / count;
                        Rect rect = transpose ? Rect{begin, end, lo, hi} : Rect{lo, hi, begin, end};
                        regions[c].push_back(rect);
                        for (int r = rect.r0; r < rect.r1; ++r) {
                            for (int col = rect.c0; col < rect.c1; ++col) {
                                for (int u : grid.cells[r * grid.cols + col]) {
                                    assignment[u] = c;
                                }
                            }
                        }
                    }
                }
                if (placements.insert(assignment).second) {
                    partitions.emplace_back(generator.placement_cost(assignment), std::move(regions));
                }
            }
        }
    }
    // 按快速代价排序矩形分区。
    std::stable_sort(partitions.begin(), partitions.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });
    if (partitions.size() > 12) {
        partitions.resize(12);
    }
    // 容量分块搜索的完整枚举参数。
    struct Config {
        int partition, rows, cols, walk, group, step;
    };
    // 容量分块阶段中，一个已完整复核的配置及其成绩。
    struct TileSeed {
        double score;
        Config config;
    };
    std::vector<TileSeed> tile_seeds;
    // 按矩形分区、块形状、遍历方式与深度步长展开序列。
    auto generate = [&](Config cfg) {
        Sequences seq(cores);
        for (int c = 0; c < cores; ++c) {
            std::vector<std::vector<int>> tiles;
            const bool transpose = cfg.walk == 1 || cfg.walk == 3;
            for (auto rect : partitions[cfg.partition].second[c]) {
                int r0 = transpose ? rect.c0 : rect.r0, r1 = transpose ? rect.c1 : rect.r1;
                int c0 = transpose ? rect.r0 : rect.c0, c1 = transpose ? rect.r1 : rect.c1;
                int dr = transpose ? cfg.cols : cfg.rows, dc = transpose ? cfg.rows : cfg.cols;
                int stripe_index = 0;
                for (int r = r0; r < r1; r += dr, ++stripe_index) {
                    std::vector<std::vector<int>> stripe;
                    for (int col = c0; col < c1; col += dc) {
                        std::vector<int> cells;
                        for (int a = r; a < std::min(r + dr, r1); ++a) {
                            for (int b = col; b < std::min(col + dc, c1); ++b) {
                                cells.push_back(transpose ? b * grid.cols + a : a * grid.cols + b);
                            }
                        }
                        stripe.push_back(std::move(cells));
                    }
                    if (cfg.walk >= 2 && stripe_index % 2) {
                        std::reverse(stripe.begin(), stripe.end());
                    }
                    for (auto &tile : stripe) {
                        tiles.push_back(std::move(tile));
                    }
                }
            }
            for (int begin = 0; begin < static_cast<int>(tiles.size()); begin += cfg.group) {
                for (int k = 0; k < grid.depth; k += cfg.step) {
                    for (int t = begin; t < std::min(begin + cfg.group, static_cast<int>(tiles.size()));
                         ++t) {
                        for (int layer = k; layer < std::min(k + cfg.step, grid.depth); ++layer) {
                            for (int cell : tiles[t]) {
                                seq[c].push_back(grid.cells[cell][layer]);
                            }
                        }
                    }
                }
            }
        }
        return seq;
    };
    std::vector<Config> pending;
    // 批量复核容量分块配置并保留有效种子。
    auto flush = [&]() {
        std::vector<Candidate> batch;
        for (auto cfg : pending) {
            batch.push_back({generate(cfg)});
        }
        prefetch(batch);
        for (std::size_t i = 0; i < pending.size(); ++i) {
            auto result = consider(batch[i].seq);
            if (result.valid) {
                tile_seeds.push_back({result.makespan, pending[i]});
            }
        }
        pending.clear();
    };
    // 缓存容量分块配置，满十个时执行批量复核。
    auto test = [&](Config cfg) {
        pending.push_back(cfg);
        if (pending.size() == 10) {
            flush();
        }
    };
    int resident = std::max(1, static_cast<int>(std::max(0.0, 0.9 * L1_CAPACITY - operand - partial) /
                                                std::max(1.0, grid.depth * operand + partial)));
    std::vector<std::pair<int, int>> initial{{4, 4}, {3, 1}, {1, 3}, {resident, 1}, {1, resident}};
    for (auto &shape : initial) {
        shape.first = std::min(shape.first, grid.rows);
        shape.second = std::min(shape.second, grid.cols);
    }
    distinct(initial);
    for (auto shape : initial) {
        for (int p = 0; p < static_cast<int>(partitions.size()); ++p) {
            test({p, shape.first, shape.second, 0, 1, grid.depth});
        }
    }
    flush();
    // 从有效配置中选出至多三种不同分区的较优配置。
    auto select = [&]() {
        // 按复核完成时间排序分块配置。
        std::stable_sort(tile_seeds.begin(), tile_seeds.end(),
                         [](const TileSeed &a, const TileSeed &b) { return a.score < b.score; });
        std::vector<Config> result;
        std::set<int> selected;
        for (const auto &seed : tile_seeds) {
            if (selected.insert(seed.config.partition).second) {
                result.push_back(seed.config);
                if (result.size() == 3) {
                    break;
                }
            }
        }
        return result;
    };
    const auto beam = select();
    std::vector<std::pair<int, int>> shapes;
    for (int r = 1; r <= std::min(12, grid.rows); ++r) {
        for (int c = 1; c <= std::min(12, grid.cols); ++c) {
            if ((r + c) * operand + (r * c + 1) * partial <= 1.15 * L1_CAPACITY) {
                shapes.emplace_back(r, c);
            }
        }
    }
    // 按边长和与面积之比排序矩形块，偏向复用效率。
    std::stable_sort(shapes.begin(), shapes.end(), [&](auto a, auto b) {
        return static_cast<double>(a.first + a.second) / (a.first * a.second) <
               static_cast<double>(b.first + b.second) / (b.first * b.second);
    });
    if (shapes.size() > 10) {
        shapes.resize(10);
    }
    for (auto shape : shapes) {
        for (auto cfg : beam) {
            for (int walk : {0, 1, 2, 3}) {
                cfg.rows = shape.first;
                cfg.cols = shape.second;
                cfg.walk = walk;
                test(cfg);
            }
        }
    }
    flush();
    const auto improved = select();
    for (auto cfg : improved) {
        for (int group : {2, 4}) {
            for (int step : {1, std::max(1, grid.depth / 2)}) {
                cfg.group = group;
                cfg.step = step;
                test(cfg);
            }
        }
    }
    flush();
}

}
}
