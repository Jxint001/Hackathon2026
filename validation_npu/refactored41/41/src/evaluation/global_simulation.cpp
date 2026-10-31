#include "evaluation/model.hpp"
namespace npu {
namespace evaluation_detail {
void advance(std::vector<Running> &running, double elapsed) {
    while (elapsed > EPS) {
        int count = 0;
        double work = std::numeric_limits<double>::infinity();
        for (const auto &r : running) {
            if (r.node >= 0 && r.ddr && r.work > EPS) {
                ++count;
                work = std::min(work, r.work);
            }
        }
        if (!count) {
            break;
        }
        double amount = std::min(work, elapsed / count);
        for (auto &r : running) {
            if (r.node >= 0 && r.ddr && r.work > EPS) {
                r.work = std::max(0.0, r.work - amount);
            }
        }
        elapsed -= amount * count;
    }
}
void project(std::vector<Running> &running, double now) {
    std::array<std::pair<double, int>, 25> active;
    int count = 0;
    for (int i = 0; i < static_cast<int>(running.size()); ++i) {
        if (running[i].node >= 0 && running[i].ddr) {
            active[count++] = {running[i].work, i};
        }
    }
    std::sort(active.begin(), active.begin() + count);
    double time = now, previous = 0;
    for (int i = 0; i < count; ++i) {
        time += (active[i].first - previous) * (count - i);
        running[active[i].second].end = std::ceil(time - EPS);
        previous = active[i].first;
    }
}
bool uses_ddr(const Task &task, int u) {
    if (!task.ops[u].copy) {
        return false;
    }
    for (int t : task.ops[u].inputs) {
        if (!task.buffers[t].memory) {
            return true;
        }
    }
    for (int t : task.ops[u].outputs) {
        if (!task.buffers[t].memory) {
            return true;
        }
    }
    return false;
}
// 同时完成的事件按核编号、PIPE_ORDER 顺序处理，然后才启动新指令。
// DDR 搬运公平共享总带宽；并发集合变化后重新投影完成时刻，保持原取整规则。
double global_replay(const std::vector<Task> &tasks, const std::vector<CrossLink> &links,
                     std::vector<double> *starts, std::vector<double> *core_ends) {
    std::vector<int> base(tasks.size() + 1);
    for (std::size_t c = 0; c < tasks.size(); ++c) {
        base[c + 1] = base[c] + static_cast<int>(tasks[c].ops.size());
    }
    int n = base.back();
    if (starts) {
        starts->assign(n, 0);
    }
    if (core_ends) {
        core_ends->assign(tasks.size(), 0);
    }
    SimulationState state(n, tasks.size() * 5);
    std::vector<double> duration(n);
    std::vector<char> ddr(n);

    std::vector<int> edge_begin(n + 1);
    for (int c = 0; c < static_cast<int>(tasks.size()); ++c) {
        for (int u = 0; u < static_cast<int>(tasks[c].ops.size()); ++u) {
            edge_begin[base[c] + u + 1] = static_cast<int>(tasks[c].ops[u].succs.size());
        }
    }
    for (const auto &link : links) {
        ++edge_begin[base[link.source_core] + link.source + 1];
    }
    std::partial_sum(edge_begin.begin(), edge_begin.end(), edge_begin.begin());
    std::vector<std::pair<int, double>> successors(edge_begin.back());
    auto edge_cursor = edge_begin;
    std::vector<std::vector<int>> fifo(tasks.size() * 5);
    for (int c = 0; c < static_cast<int>(tasks.size()); ++c) {
        const auto &task = tasks[c];
        for (int u = 0; u < static_cast<int>(task.ops.size()); ++u) {
            int global = base[c] + u;
            duration[global] = task.ops[u].duration;
            ddr[global] = uses_ddr(task, u);
            state.remaining[global] = static_cast<int>(task.ops[u].preds.size());
            for (int v : task.ops[u].succs) {
                successors[edge_cursor[global]++] = {base[c] + v, 0};
            }
        }
        for (int pi = 0; pi < 5; ++pi) {
            for (int u : task.fifo[PIPE_ORDER[pi]]) {
                fifo[c * 5 + pi].push_back(base[c] + u);
            }
        }
    }
    for (const auto &link : links) {
        int u = base[link.source_core] + link.source, v = base[link.target_core] + link.target;
        successors[edge_cursor[u]++] = {v, CROSS_CORE_DELAY};
        ++state.remaining[v];
    }
    while (state.done < n) {
        advance(state.running, state.now - state.last);
        state.last = state.now;
        for (std::size_t p = 0; p < fifo.size(); ++p) {
            if (state.running[p].node >= 0 && state.running[p].end <= state.now + EPS) {
                int u = state.running[p].node;
                if (core_ends) {
                    (*core_ends)[p / 5] = state.now;
                }
                for (int j = edge_begin[u]; j < edge_begin[u + 1]; ++j) {
                    const auto &edge = successors[j];
                    --state.remaining[edge.first];
                    state.release[edge.first] = std::max(state.release[edge.first], state.now + edge.second);
                }
                state.running[p].node = -1;
                ++state.cursor[p];
                ++state.done;
            }
        }
        for (std::size_t p = 0; p < fifo.size(); ++p) {
            if (state.running[p].node >= 0 || state.cursor[p] == fifo[p].size()) {
                continue;
            }
            int u = fifo[p][state.cursor[p]];
            if (state.remaining[u] || state.release[u] > state.now) {
                continue;
            }
            if (starts) {
                (*starts)[u] = state.now;
            }
            state.running[p] = {u, state.now + duration[u], duration[u], ddr[u] != 0};
        }
        if (state.done == n) {
            return state.now;
        }
        project(state.running, state.now);
        double next = std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < fifo.size(); ++p) {
            if (state.running[p].node >= 0) {
                next = std::min(next, state.running[p].end);
            } else if (state.cursor[p] < fifo[p].size()) {
                int u = fifo[p][state.cursor[p]];
                if (!state.remaining[u] && state.release[u] > state.now) {
                    next = std::min(next, state.release[u]);
                }
            }
        }
        if (!std::isfinite(next) || next <= state.now) {
            throw InvalidReview{};
        }
        state.now = next;
    }
    return state.now;
}
}
}
