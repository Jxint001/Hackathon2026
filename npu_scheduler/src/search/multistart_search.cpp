#include "thread_pool.hpp"
#include "search/common.hpp"
#include "search/search_controller.hpp"
namespace npu {
namespace search_detail {
void SearchController::alternate_seeds() {
    if (state.seeds.empty()) {
        return;
    }
    std::vector<int> order, stack;
    std::vector<char> seen(size);
    // 为计算图逆向 DFS 比较操作深度与 id。
    auto less = [&](int a, int b) {
        return std::make_tuple(generator.depth[a], -problem.ops[a].id) <
               std::make_tuple(generator.depth[b], -problem.ops[b].id);
    };
    for (int u : generator.topo) {
        if (problem.ops[u].succs.empty()) {
            stack.push_back(u);
        }
    }
    std::sort(stack.begin(), stack.end(), less);
    while (!stack.empty()) {
        int u = stack.back();
        if (seen[u]) {
            stack.pop_back();
            continue;
        }
        std::vector<int> pending;
        for (int p : problem.ops[u].preds) {
            if (!seen[p]) {
                pending.push_back(p);
            }
        }
        if (pending.empty()) {
            seen[u] = true;
            order.push_back(u);
            stack.pop_back();
        } else {
            std::sort(pending.begin(), pending.end(), less);
            stack.insert(stack.end(), pending.begin(), pending.end());
        }
    }
    auto ranked = state.seeds;
    // 按完成时间对已有空间起点稳定排序。
    std::stable_sort(ranked.begin(), ranked.end(), [](const SearchSeed &a, const SearchSeed &b) {
        return a.score.makespan < b.score.makespan;
    });
    std::set<std::vector<int>> placements;
    std::vector<double> durations;
    for (int u : generator.topo) {
        if (problem.ops[u].pipe == 0 && problem.ops[u].cycles > 1) {
            durations.push_back(problem.ops[u].cycles);
        }
    }
    const double bias = 2.0 * median(durations);
    std::vector<Candidate> batch;
    for (const auto &seed : ranked) {
        if (!placements.insert(seed.assignment).second) {
            continue;
        }
        auto reference = generator.sequences_for_assignment(seed.assignment, order);
        for (int window : {64, 4096}) {
            for (double preference : {0.0, bias}) {
                auto seq = generator.window_order(reference, window, preference);
                int family = 4 + (window == 4096 ? 2 : 0) + (preference > 0);
                batch.push_back({seq, 1, family});
                batch.push_back({generator.delay_producers(std::move(seq), 1, true), 1, family});
            }
        }
        if (placements.size() == 4) {
            break;
        }
    }
    std::vector<PlanEvaluation> scores(batch.size());
    // 并行复核替代空间排序起点。
    parallel_for(batch.size(), generator.topo.size() >= 256,
                 [&](std::size_t i) { scores[i] = evaluator.evaluate(batch[i]); });
    for (std::size_t i = 0; i < batch.size(); ++i) {
        ++state.total_attempts;
        remember_seed(batch[i], scores[i]);
    }
}

void SearchController::multistart_search(std::uint64_t starting_hash) {
    if (state.seeds.empty()) {
        return;
    }

    const double promotion_limit = state.best.score.makespan * 0.995;
    alternate_seeds();
    // 按完成时间、搬运量排列空间起点。
    std::stable_sort(state.seeds.begin(), state.seeds.end(), [](const SearchSeed &a, const SearchSeed &b) {
        return std::tie(a.score.makespan, a.score.copy_bytes) <
               std::tie(b.score.makespan, b.score.copy_bytes);
    });
    std::map<int, int> per_family;
    // 限制每种排序族保留的起点数量。
    state.seeds.erase(
        std::remove_if(state.seeds.begin(), state.seeds.end(),
                       [&](const SearchSeed &seed) { return per_family[seed.candidate.family]++ >= 4; }),
        state.seeds.end());

    for (auto &seed : state.seeds) {
        auto s = evaluator.materialize_solution(seed.candidate.seq, 1);
        s.score.makespan = seed.score.makespan;
        s.score.copy_bytes = seed.score.copy_bytes;
        SearchController local(problem, cores, s);
        copy_completed_to(local);
        local.phase(SearchStage::DelayProducers);
        local.copy_completed_to(*this);
        seed.candidate.seq = local.trace_best().sequences;
        seed.score = {true, local.state.best.score.makespan, local.state.best.score.copy_bytes, 0};
        state.total_attempts += local.state.total_attempts;
    }
    // 按短搜索后的完成时间和搬运量重新排序起点。
    std::stable_sort(state.seeds.begin(), state.seeds.end(), [](const SearchSeed &a, const SearchSeed &b) {
        return std::tie(a.score.makespan, a.score.copy_bytes) <
               std::tie(b.score.makespan, b.score.copy_bytes);
    });
    std::vector<SearchSeed> selected;
    std::set<int> families;
    std::set<std::vector<int>> placements;
    for (const auto &seed : state.seeds) {
        if (families.insert(seed.candidate.family).second) {
            selected.push_back(seed);
            placements.insert(seed.assignment);
            if (selected.size() == 4) {
                break;
            }
        }
    }
    for (const auto &seed : state.seeds) {
        if (placements.insert(seed.assignment).second) {
            selected.push_back(seed);
            if (selected.size() == 6) {
                break;
            }
        }
    }
    state.seeds.clear();
    std::set<std::uint64_t> starts{starting_hash};
    for (const auto &seed : selected) {
        if (!starts.insert(evaluator.candidate_hash(seed.candidate.seq, 1)).second) {
            continue;
        }
        auto s = evaluator.materialize_solution(seed.candidate.seq, seed.candidate.chunk);
        s.score.makespan = seed.score.makespan;
        s.score.copy_bytes = seed.score.copy_bytes;
        SearchController local(problem, cores, s);
        copy_completed_to(local);
        local.phase(SearchStage::ConvolutionOrder);
        local.copy_completed_to(*this);
        auto solution = local.finish();
        state.total_attempts += solution.statistics.evaluations;
        if (solution.score.makespan <= promotion_limit &&
            better({true, solution.score.makespan, solution.score.copy_bytes, 0})) {
            solution.statistics.evaluations = state.best.statistics.evaluations;
            state.best = std::move(solution);
            state.best_trace.reset();
            ++state.best_revision;
        }
    }
}

void SearchController::matrix_starts() {
    if (state.best.score.makespan <= generator.resource_lower_bound(cores) * 1.01) {
        return;
    }
    const auto grid = generator.recognize_grid();
    if (!grid.rows) {
        return;
    }
    for (int u : generator.topo) {
        if (problem.ops[u].pipe != 0) {
            return;
        }
    }
    // 矩阵面板的矩形范围，在当前选定的遍历坐标系中表示。
    struct Panel {
        int r0, r1, c0, c1;
    };
    // 矩阵多起点搜索中的面板布局，与条带搜索的 Layout 别名作用域不同。
    struct Layout {
        bool transpose = false;
        std::vector<std::vector<Panel>> parts;
    };
    // 展开面板布局，沿用前一面板末列以增加边界复用。
    auto sequence = [&](const Layout &layout) {
        Sequences seq(cores);
        for (int c = 0; c < cores; ++c) {
            int previous = -1;
            for (const auto &p : layout.parts[c]) {
                const int width = p.c1 - p.c0;
                int first = p.c0;
                if (previous >= 0) {
                    first = std::clamp(previous, p.c0, p.c1 - 1);
                }
                for (int j = 0; j < width; ++j) {
                    int col = p.c0 + (first - p.c0 + j) % width;
                    for (int k = 0; k < grid.depth; ++k) {
                        for (int row = p.r0; row < p.r1; ++row) {
                            int cell = layout.transpose ? col * grid.cols + row : row * grid.cols + col;
                            seq[c].push_back(grid.cells[cell][k]);
                        }
                    }
                    previous = col;
                }
            }
        }
        return seq;
    };
    std::vector<Layout> pending;
    std::vector<SearchSeed> starts;
    // 批量复核面板起点并保存有效成绩。
    auto flush = [&]() {
        std::vector<Candidate> batch;
        for (const auto &layout : pending) {
            batch.push_back({sequence(layout)});
        }
        prefetch(batch);
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const auto r = consider(batch[i].seq);
            if (r.valid) {
                starts.push_back({batch[i], r, generator.core_assignment(batch[i].seq)});
            }
        }
        pending.clear();
    };
    // 缓存一个面板布局，十个一批送入复核。
    auto test = [&](Layout layout) {
        pending.push_back(std::move(layout));
        if (pending.size() == 10) {
            flush();
        }
    };
    for (int active = cores; active >= 2; --active) {
        if (generator.resource_lower_bound(active) > state.best.score.makespan) {
            continue;
        }
        for (int height : {3, 2, 4, 1, 6, 8}) {
            for (bool transpose : {false, true}) {
                for (bool tail_first : {true, false}) {
                    int rows = transpose ? grid.cols : grid.rows, columns = transpose ? grid.rows : grid.cols;
                    if (height > rows) {
                        continue;
                    }
                    int per_core = rows / (active * height) * height, tail = per_core * active;
                    Layout layout;
                    layout.transpose = transpose;
                    layout.parts.resize(cores);
                    for (int c = 0; c < active; ++c) {
                        auto &panels = layout.parts[c];
                        for (int r = c * per_core; r < (c + 1) * per_core; r += height) {
                            panels.push_back({r, r + height, 0, columns});
                        }
                        int lo = (c * columns + active - 1) / active,
                            hi = ((c + 1) * columns + active - 1) / active;
                        if (tail < rows && lo < hi) {
                            Panel last{tail, rows, lo, hi};
                            if (tail_first) {
                                panels.insert(panels.begin(), last);
                            } else {
                                panels.push_back(last);
                            }
                        }
                    }
                    test(std::move(layout));
                }
            }
        }
    }
    flush();

    // 按复核时间和搬运量排列矩阵起点。
    std::stable_sort(starts.begin(), starts.end(), [](const SearchSeed &a, const SearchSeed &b) {
        return std::tie(a.score.makespan, a.score.copy_bytes) <
               std::tie(b.score.makespan, b.score.copy_bytes);
    });
    std::map<std::vector<int>, int> placements;
    int count = 0;
    for (const auto &seed : starts) {
        if (placements[seed.assignment]++ >= 2) {
            continue;
        }
        if (count++ == 6) {
            break;
        }
        auto s = evaluator.materialize_solution(seed.candidate.seq, 1);
        s.score.makespan = seed.score.makespan;
        s.score.copy_bytes = seed.score.copy_bytes;

        int active = *std::max_element(seed.assignment.begin(), seed.assignment.end()) + 1;
        SearchController local(problem, active, s);
        copy_completed_to(local);
        local.phase(SearchStage::DelayProducers);
        local.phase(SearchStage::ChainMoves);
        local.copy_completed_to(*this);
        auto solution = local.finish();
        state.total_attempts += solution.statistics.evaluations;
        if (better({true, solution.score.makespan, solution.score.copy_bytes, 0})) {
            solution.statistics.evaluations = state.best.statistics.evaluations;
            state.best = std::move(solution);
            state.best_trace.reset();
            ++state.best_revision;
        }
    }
}

}
}
