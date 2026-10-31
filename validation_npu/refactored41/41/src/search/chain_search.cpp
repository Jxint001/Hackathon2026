#include "thread_pool.hpp"
#include "search/common.hpp"
#include "search/search_controller.hpp"
namespace npu {
namespace search_detail {
void SearchController::chain_moves() {
    const auto grid = generator.recognize_grid();
    if (!grid.rows || state.best.score.makespan <= generator.resource_lower_bound(cores) * 1.01) {
        return;
    }
    for (int u : generator.topo) {
        if (problem.ops[u].pipe != 0) {
            return;
        }
    }
    // 一次迁核操作涉及的（操作索引，目标核心编号）列表。
    using Changes = std::vector<std::pair<int, int>>;
    for (int round = 0; round < 6 && state.phase_attempts < state.phase_limit; ++round) {
        double before = state.best.score.makespan;
        auto trace = trace_best();
        const auto a = generator.core_assignment(trace.sequences);
        auto assignment = a;
        std::vector<int> destinations(cores);
        std::iota(destinations.begin(), destinations.end(), 0);
        // 让较早完成的核心优先作为迁移目标。
        std::stable_sort(destinations.begin(), destinations.end(),
                         [&](int x, int y) { return trace.core_ends[x] < trace.core_ends[y]; });
        auto order = generator.topo;
        // 按当前执行开始时间重建统一参考顺序。
        std::stable_sort(order.begin(), order.end(), [&](int u, int v) {
            return std::tie(trace.starts[u], u) < std::tie(trace.starts[v], v);
        });

        std::vector<std::array<int, 5>> readers(problem.tensors.size());
        std::array<double, 5> work{};
        for (int u : generator.topo) {
            work[a[u]] += problem.ops[u].cycles;
            for (int t : problem.ops[u].inputs) {
                ++readers[t][a[u]];
            }
        }
        // 计算当前临时分核下单张量的必需搬运周期。
        auto traffic = [&](int t) {
            const auto &tensor = problem.tensors[t];
            int copies = 0;
            if (tensor.producers.empty()) {
                for (int c = 0; c < cores; ++c) {
                    copies += readers[t][c] > 0;
                }
            } else {
                for (int c = 0; c < cores; ++c) {
                    if (c != assignment[tensor.producers.front()]) {
                        copies += 2 * (readers[t][c] > 0);
                    }
                }
                copies += tensor.output;
            }
            return copies * std::max(1.0, std::ceil(tensor.size / DDR_BANDWIDTH));
        };
        double base_ddr = 0;
        for (int t = 0; t < static_cast<int>(readers.size()); ++t) {
            base_ddr += traffic(t);
        }
        // 增量预筛选之后保留的链迁移提案。
        struct Move {
            double cost;
            Changes changes;
        };
        std::array<std::vector<Move>, 4> pools;
        // 闭合共享写者约束并增量预筛选迁核候选。
        auto offer = [&](Changes changes, int kind) {
            std::map<int, int> closed;
            for (auto change : changes) {
                for (int u : generator.members[generator.atom[change.first]]) {
                    auto old = closed.find(u);
                    if (old != closed.end() && old->second != change.second) {
                        return;
                    }
                    closed[u] = change.second;
                }
            }
            changes.clear();
            for (auto change : closed) {
                if (a[change.first] != change.second) {
                    changes.push_back(change);
                }
            }
            if (changes.empty()) {
                return;
            }
            std::vector<int> touched;
            for (auto change : changes) {
                const auto &op = problem.ops[change.first];
                touched.insert(touched.end(), op.inputs.begin(), op.inputs.end());
                touched.insert(touched.end(), op.outputs.begin(), op.outputs.end());
            }
            distinct(touched);
            double ddr = base_ddr;
            for (int t : touched) {
                ddr -= traffic(t);
            }
            for (auto change : changes) {
                int u = change.first, c = change.second;
                const auto &op = problem.ops[u];
                work[a[u]] -= op.cycles;
                work[c] += op.cycles;
                assignment[u] = c;
                for (int t : op.inputs) {
                    --readers[t][a[u]];
                    ++readers[t][c];
                }
            }
            for (int t : touched) {
                ddr += traffic(t);
            }
            double peak = *std::max_element(work.begin(), work.end());
            for (auto change : changes) {
                int u = change.first, c = change.second;
                const auto &op = problem.ops[u];
                work[a[u]] += op.cycles;
                work[c] -= op.cycles;
                assignment[u] = a[u];
                for (int t : op.inputs) {
                    ++readers[t][a[u]];
                    --readers[t][c];
                }
            }
            if (std::max(peak, ddr) > state.best.score.makespan) {
                return;
            }
            pools[kind].push_back({std::max(peak, ddr) + .15 * ddr + .05 * peak, std::move(changes)});
        };
        for (const auto &chain : grid.cells) {
            for (int target : destinations) {
                for (int part = 0; part < 3; ++part) {
                    if (part && chain.size() < 2) {
                        continue;
                    }
                    Changes changes;
                    int lo = part == 2 ? static_cast<int>(chain.size() / 2) : 0;
                    int hi = part == 1 ? static_cast<int>(chain.size() / 2) : static_cast<int>(chain.size());
                    for (int k = lo; k < hi; ++k) {
                        changes.emplace_back(chain[k], target);
                    }
                    offer(std::move(changes), part);
                }
            }
            if (chain.size() > 1 && cores > 2) {
                std::vector<int> targets;
                for (int c : destinations) {
                    if (c != a[chain.front()]) {
                        targets.push_back(c);
                    }
                }
                for (int reverse = 0; reverse < 2; ++reverse) {
                    Changes changes;
                    for (std::size_t k = 0; k < chain.size(); ++k) {
                        changes.emplace_back(chain[k], targets[(k >= chain.size() / 2) ^ reverse]);
                    }
                    offer(std::move(changes), 3);
                }
            }
        }
        std::vector<Candidate> batch;
        for (auto &pool : pools) {
            // 按快速代价和迁移列表稳定排序候选。
            std::stable_sort(pool.begin(), pool.end(), [](const Move &x, const Move &y) {
                return std::tie(x.cost, x.changes) < std::tie(y.cost, y.changes);
            });
            if (pool.size() > 8) {
                pool.resize(8);
            }
            for (const auto &move : pool) {
                assignment = a;
                for (auto change : move.changes) {
                    assignment[change.first] = change.second;
                }
#ifndef NDEBUG
                const auto check = generator.placement_resources(assignment);
                assert(std::abs(move.cost - (std::max(check.peak, check.ddr) + .15 * check.ddr +
                                             .05 * check.total_peak)) < 1e-6);
#endif
                auto reference = generator.sequences_for_assignment(assignment, order);
                batch.push_back({generator.delay_producers(reference, 1, false)});
                batch.push_back(
                    {generator.delay_producers(generator.window_order(reference, 512, 0), 1, true)});
            }
        }
        consider_batch(batch);
        if (state.best.score.makespan >= before - .5) {
            break;
        }
    }
}

}
}
