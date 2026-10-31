#include "thread_pool.hpp"
#include "search/common.hpp"
#include "search/search_controller.hpp"
namespace npu {
namespace search_detail {
void SearchController::overlap_search() {
    const double incumbent = state.best.score.makespan;
    const auto coord = locality_axis(problem);
    const double maximum = *std::max_element(coord.begin(), coord.end());
    if (maximum < 0) {
        return;
    }
    std::map<int, std::vector<double>> layers;
    for (int u : generator.topo) {
        if (coord[u] >= 0) {
            layers[generator.depth[u]].push_back(coord[u]);
        }
    }
    std::set<std::vector<int>> seen;
    std::vector<std::pair<double, std::vector<int>>> candidates;
    // 去重并登记空间分核候选的快速代价。
    auto offer = [&](std::vector<int> a) {
        if (seen.insert(a).second) {
            candidates.emplace_back(generator.placement_cost(a), std::move(a));
        }
    };
    for (int active = cores; active >= 2; --active) {
        if (generator.resource_lower_bound(active) > state.best.score.makespan) {
            continue;
        }
        for (bool multi : {false, true}) {
            offer(generator.coordinate_assignment(coord, active, multi));
        }
        {
            std::vector<std::pair<double, int>> ordered;
            double total_work = 0, accumulated = 0;
            for (int v = 0; v < size; ++v) {
                if (!generator.members[v].empty()) {
                    double value = 0;
                    for (int u : generator.members[v]) {
                        value += coord[u];
                    }
                    ordered.emplace_back(value / generator.members[v].size(), v);
                    total_work += std::accumulate(generator.loads[v].begin(), generator.loads[v].end(), 0.0);
                }
            }
            std::sort(ordered.begin(), ordered.end());
            std::vector<int> a(size, -1);
            for (auto entry : ordered) {
                int v = entry.second;
                double work = std::accumulate(generator.loads[v].begin(), generator.loads[v].end(), 0.0);
                int c = std::min(active - 1, static_cast<int>((accumulated + work / 2) * active /
                                                              std::max(1.0, total_work)));
                for (int u : generator.members[v]) {
                    a[u] = c;
                }
                accumulated += work;
            }
            offer(std::move(a));
        }
        for (double shift : {0.0, 0.25}) {
            std::vector<int> a(size, -1);
            for (int v = 0; v < size; ++v) {
                if (!generator.members[v].empty()) {
                    double value = 0;
                    for (int u : generator.members[v]) {
                        value += coord[u];
                    }
                    int c = std::clamp(static_cast<int>((value / generator.members[v].size() + 0.5 + shift) *
                                                        active / (maximum + 1)),
                                       0, active - 1);
                    for (int u : generator.members[v]) {
                        a[u] = c;
                    }
                }
            }
            offer(std::move(a));
        }
        double step = 1;
        for (const auto &layer : layers) {
            auto values = layer.second;
            distinct(values);
            if (values.size() < static_cast<std::size_t>(2 * active)) {
                continue;
            }
            std::vector<double> gaps;
            for (std::size_t i = 1; i < values.size(); ++i) {
                gaps.push_back(values[i] - values[i - 1]);
            }
            double spacing = median(gaps);
            int regular = 0;
            for (double gap : gaps) {
                regular += std::abs(gap - spacing) < 0.01 * spacing;
            }
            if (regular * 4 >= static_cast<int>(gaps.size()) * 3) {
                step = std::max(step, spacing);
            }
        }
        const int cells = std::max(active, static_cast<int>(std::ceil((maximum + 1) / step)));
        std::vector<int> widths(active, cells / active);
        for (int i = 0; i < cells % active; ++i) {
            ++widths[active - 1 - i];
        }
        do {
            std::vector<int> ends(active), a(size, -1);
            std::partial_sum(widths.begin(), widths.end(), ends.begin());
            for (int v = 0; v < size; ++v) {
                if (!generator.members[v].empty()) {
                    double value = 0;
                    for (int u : generator.members[v]) {
                        value += coord[u];
                    }
                    int cell = std::max(0, static_cast<int>(value / generator.members[v].size() / step));
                    int c = std::min(
                        active - 1,
                        static_cast<int>(std::upper_bound(ends.begin(), ends.end(), cell) - ends.begin()));
                    for (int u : generator.members[v]) {
                        a[u] = c;
                    }
                }
            }
            offer(std::move(a));
        } while (std::next_permutation(widths.begin(), widths.end()));
    }
    {
        auto variants = candidates;
        for (auto &candidate : variants) {
            auto a = std::move(candidate.second);
            for (int u : generator.topo) {
                if (problem.ops[u].preds.empty() &&
                    problem.ops[u].succs.size() >= static_cast<std::size_t>(4 * cores) &&
                    problem.ops[u].cycles <= CROSS_CORE_DELAY / 8) {
                    for (int v : generator.members[generator.atom[u]]) {
                        a[v] = 0;
                    }
                }
            }
            offer(std::move(a));
        }
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<double> durations;
    for (int u : generator.topo) {
        if (problem.ops[u].pipe == 0 && problem.ops[u].cycles > 1) {
            durations.push_back(problem.ops[u].cycles);
        }
    }
    const double bias = durations.empty() ? 0 : 2.5 * median(durations);
    std::vector<Candidate> batch;
    state.collect_seeds =
        generator.total[0] && std::accumulate(generator.total.begin() + 1, generator.total.end(), 0.0) > 0;
    for (int i = 0; i < std::min(12, static_cast<int>(candidates.size())); ++i) {
        const auto &a = candidates[i].second;
        {
            const auto reference = generator.sequences_for_assignment(a, generator.dfs);
            for (int window : {64, 4096}) {
                for (double preference : {0.0, bias}) {
                    auto seq = generator.window_order(reference, window, preference);
                    int family = (window == 4096 ? 2 : 0) + (preference > 0);
                    batch.push_back({seq, 1, family});
                    batch.push_back({generator.delay_producers(std::move(seq), 1, true), 1, family});
                    if (batch.size() == 10) {
                        consider_batch(batch);
                        batch.clear();
                    }
                }
            }
        }
    }
    consider_batch(batch);
    state.collect_seeds = false;
    if (state.best.score.makespan < incumbent - 0.5) {
        producer_search();
    }
}

}
}
