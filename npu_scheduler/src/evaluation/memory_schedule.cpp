#include "evaluation/model.hpp"
namespace npu {
namespace evaluation_detail {
struct Spill {
    int tensor, previous, next;
};
double insert_spills(Task &task) {
    const int n = static_cast<int>(task.sequence.size());
    const int nt = static_cast<int>(task.buffers.size());
    if (!n) {
        return 0;
    }
    std::pmr::monotonic_buffer_resource scratch;
    std::vector<int> position(task.ops.size());
    for (int i = 0; i < n; ++i) {
        position[task.sequence[i]] = i;
    }
    std::pmr::vector<Indices> uses(nt, &scratch), writes(nt, &scratch);
    std::vector<int> at_begin(n + 1);
    for (int t = 0; t < nt; ++t) {
        if (task.buffers[t].memory) {
            for (int u : task.buffers[t].writers) {
                uses[t].push_back(position[u]);
                writes[t].push_back(position[u]);
            }
            for (int u : task.buffers[t].readers) {
                uses[t].push_back(position[u]);
            }
            distinct(uses[t]);
            distinct(writes[t]);
            for (int step : uses[t]) {
                ++at_begin[step + 1];
            }
        }
    }
    std::partial_sum(at_begin.begin(), at_begin.end(), at_begin.begin());
    std::vector<std::pair<int, int>> at(at_begin.back());
    auto at_cursor = at_begin;
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < static_cast<int>(uses[t].size()); ++i) {
            at[at_cursor[uses[t][i]]++] = {t, i};
        }
    }
    // 精细换出计划中，一个缓冲区当前的驻留与未来使用信息。
    struct Active {
        bool resident = false;
        int next = 0, used = 0, stamp = 0;
    };
    std::vector<Active> active(nt);

    // 容器元素类型：换出优先队列中的三元组。
    using Entry = std::tuple<int, int, int>;
    std::array<std::priority_queue<Entry>, 3> heaps;
    std::array<Id, 3> resident{};
    std::pmr::vector<Indices> reload(n, &scratch);
    std::vector<Spill> spills;
    std::vector<char> pinned(nt);
    int serial = 0;
    for (int step = 0; step < n; ++step) {
        for (int t : reload[step]) {
            active[t].resident = true;
            active[t].stamp = ++serial;
            resident[task.buffers[t].memory] += task.buffers[t].size;
        }
        for (int j = at_begin[step]; j < at_begin[step + 1]; ++j) {
            const auto &use = at[j];
            int t = use.first, i = use.second;
            auto &state = active[t];
            pinned[t] = true;
            if (i == 0) {
                state.resident = true;
                state.stamp = ++serial;
                resident[task.buffers[t].memory] += task.buffers[t].size;
            }
            state.used = i + 1;
            state.next = i + 1 < static_cast<int>(uses[t].size()) ? uses[t][i + 1] : n;
            heaps[task.buffers[t].memory].emplace(state.next, -state.stamp, t);
        }
        for (int mem = 1; mem <= 2; ++mem) {
            auto &heap = heaps[mem];
            while (resident[mem] > CAPACITY[mem] * 9 / 10) {
                int victim = -1;
                while (!heap.empty()) {
                    auto entry = heap.top();
                    heap.pop();
                    int t = std::get<2>(entry);
                    const auto &a = active[t];
                    if (!a.resident || a.next != std::get<0>(entry) || -a.stamp != std::get<1>(entry) ||
                        pinned[t] || a.next == n) {
                        continue;
                    }
                    victim = t;
                    break;
                }
                if (victim < 0) {
                    if (resident[mem] > CAPACITY[mem]) {
                        throw InvalidReview{};
                    }
                    break;
                }
                auto &a = active[victim];
                spills.push_back({victim, uses[victim][a.used - 1], a.next});
                reload[a.next].push_back(victim);
                a.resident = false;
                resident[mem] -= task.buffers[victim].size;
            }
        }
        for (int j = at_begin[step]; j < at_begin[step + 1]; ++j) {
            const auto &use = at[j];
            int t = use.first;
            auto &a = active[t];
            pinned[t] = false;
            if (a.next == n) {
                a.resident = false;
                resident[task.buffers[t].memory] -= task.buffers[t].size;
            } else {
                heaps[task.buffers[t].memory].emplace(a.next, -a.stamp, t);
            }
        }
    }

    if (spills.empty()) {
        return 0;
    }
    Key next_id = 0;
    for (const auto &op : task.ops) {
        next_id = std::max(next_id, op.id + 1);
    }
    for (const auto &t : task.buffers) {
        next_id = std::max(next_id, t.id + 1);
    }
    std::vector<int> backing(nt, -1), synced(nt, -1), current(nt);
    std::iota(current.begin(), current.end(), 0);
    for (int t = 0; t < nt; ++t) {
        if (task.buffers[t].memory) {
            std::vector<int> sources;
            for (int u : task.buffers[t].writers) {
                if (task.ops[u].copy == 1) {
                    std::vector<int> ddr;
                    for (int input : task.ops[u].inputs) {
                        if (!task.buffers[input].memory) {
                            ddr.push_back(input);
                        }
                    }
                    if (ddr.size() == 1) {
                        sources.push_back(ddr.front());
                    }
                }
            }
            distinct(sources);
            if (sources.size() == 1) {
                backing[t] = sources.front();
                synced[t] = uses[t].front();
            }
        }
    }
    std::pmr::vector<std::pmr::vector<std::pair<int, int>>> versions(nt, &scratch);
    std::pmr::vector<Indices> before(n, &scratch), after(n, &scratch);
    double traffic = 0;
    for (const auto &sp : spills) {
        int t = sp.tensor, from = current[t];
        auto writer = std::upper_bound(writes[t].begin(), writes[t].end(), synced[t]);
        bool dirty = backing[t] < 0 || (writer != writes[t].end() && *writer <= sp.previous);
        Key out_id = dirty ? next_id++ : 0;
        Key in_id = next_id++;
        const auto size = task.buffers[t].size;
        const auto memory = task.buffers[t].memory;
        if (dirty) {
            backing[t] = buffer(task, next_id++, size, 0);
            synced[t] = sp.previous;
        }
        int to = buffer(task, next_id++, size, memory);
        if (dirty) {
            int out = copy(task, out_id, 2, from, backing[t], -1);
            after[sp.previous].push_back(out);
            traffic += size;
        }
        int in = copy(task, in_id, 1, backing[t], to, -1);
        before[sp.next].push_back(in);
        traffic += size;
        current[t] = to;
        versions[t].emplace_back(sp.next, to);
    }

    for (int u = 0; u < n; ++u) {
        // 将原指令的输入或输出替换为当前时刻有效的缓冲区实例。
        auto rewrite = [&](auto &tensors) {
            for (int &t : tensors) {
                const auto &vs = versions[t];
                auto it = std::upper_bound(vs.begin(), vs.end(),
                                           std::make_pair(position[u], std::numeric_limits<int>::max()));
                if (it != vs.begin()) {
                    t = std::prev(it)->second;
                }
            }
        };
        rewrite(task.ops[u].inputs);
        rewrite(task.ops[u].outputs);
    }
    std::vector<int> sequence;
    for (int i = 0; i < n; ++i) {
        sequence.insert(sequence.end(), before[i].begin(), before[i].end());
        sequence.push_back(task.sequence[i]);
        sequence.insert(sequence.end(), after[i].begin(), after[i].end());
    }
    task.sequence = std::move(sequence);
    build_views(task);
    position.resize(task.ops.size());
    for (int i = 0; i < static_cast<int>(task.sequence.size()); ++i) {
        position[task.sequence[i]] = i;
    }
    for (int u = 0; u < static_cast<int>(task.ops.size()); ++u) {
        for (int p : task.ops[u].preds) {
            if (position[p] >= position[u]) {
                throw InvalidReview{};
            }
        }
    }
    return traffic;
}
// 先完成事件释放容量，再按提交的申请顺序分配；不越过硬容量。
void prepare_memory(Task &task) {
    int n = static_cast<int>(task.ops.size()), nt = static_cast<int>(task.buffers.size());
    if (!n) {
        return;
    }
    std::array<std::size_t, 5> cursor{};
    std::vector<int> allocation, remaining(n), outputs;
    MemoryState memory(nt);
    for (int u : task.sequence) {
        task.fifo[task.ops[u].pipe].push_back(u);
        bool allocates = false;
        for (int t : task.ops[u].outputs) {
            allocates |= task.buffers[t].memory != 0;
        }
        if (allocates) {
            allocation.push_back(u);
        }
        remaining[u] = static_cast<int>(task.ops[u].preds.size());
    }
    // 核内容量空闲队列中的一段可复用容量及其释放来源。
    for (int t = 0; t < nt; ++t) {
        memory.refs[t] = static_cast<int>(task.buffers[t].readers.size());
        if (task.buffers[t].memory && task.buffers[t].writers.empty() && memory.refs[t]) {
            memory.resident[t] = true;
            memory.used[task.buffers[t].memory] += task.buffers[t].size;
        }
    }
    for (int m = 1; m <= 2; ++m) {
        if (memory.used[m] > CAPACITY[m]) {
            throw InvalidReview{};
        }
        memory.credits[m].push_back({CAPACITY[m] - memory.used[m], -1});
    }
    // 释放不再使用的驻留缓冲区并记录可复用容量。
    auto release = [&](int t) {
        const auto &b = task.buffers[t];
        if (!b.memory || !memory.resident[t]) {
            return;
        }
        memory.resident[t] = false;
        memory.used[b.memory] -= b.size;
        memory.credits[b.memory].push_back({b.size, t});
    };
    std::vector<Running> running(5);
    double now = 0, last = 0;
    int done = 0;
    while (done < n) {
        advance(running, now - last);
        last = now;
        for (int p : PIPE_ORDER) {
            if (running[p].node >= 0 && running[p].end <= now + EPS) {
                int u = running[p].node;
                for (int t : task.ops[u].inputs) {
                    if (task.buffers[t].memory && !--memory.refs[t]) {
                        release(t);
                    }
                }
                for (int t : task.ops[u].outputs) {
                    if (task.buffers[t].memory && !memory.refs[t]) {
                        release(t);
                    }
                }
                running[p].node = -1;
                ++cursor[p];
                ++done;
                for (int v : task.ops[u].succs) {
                    --remaining[v];
                }
            }
        }
        bool progress = true;
        while (progress) {
            progress = false;
            for (int p : PIPE_ORDER) {
                if (running[p].node >= 0 || cursor[p] == task.fifo[p].size()) {
                    continue;
                }
                int u = task.fifo[p][cursor[p]];
                if (remaining[u]) {
                    continue;
                }
                bool allocates = false;
                std::array<Id, 3> need{};
                for (int t : task.ops[u].outputs) {
                    if (task.buffers[t].memory) {
                        allocates = true;
                        if (!memory.resident[t]) {
                            need[task.buffers[t].memory] += task.buffers[t].size;
                        }
                    }
                }
                if (allocates && (memory.allocation_cursor == allocation.size() ||
                                  allocation[memory.allocation_cursor] != u)) {
                    continue;
                }
                if (memory.used[1] + need[1] > CAPACITY[1] || memory.used[2] + need[2] > CAPACITY[2]) {
                    continue;
                }
                if (allocates) {
                    ++memory.allocation_cursor;
                }
                outputs.assign(task.ops[u].outputs.begin(), task.ops[u].outputs.end());
                // 按缓冲区标识固定内存申请顺序。
                std::sort(outputs.begin(), outputs.end(),
                          [&](int a, int b) { return task.buffers[a].id < task.buffers[b].id; });
                for (int t : outputs) {
                    const auto &b = task.buffers[t];
                    if (!b.memory || memory.resident[t]) {
                        continue;
                    }
                    Id bytes = b.size;
                    auto &pool = memory.credits[b.memory];
                    while (bytes > 0 && !pool.empty()) {
                        auto &credit = pool.front();
                        Id take = std::min(bytes, credit.bytes);
                        if (credit.tensor >= 0) {
                            const auto &previous = task.buffers[credit.tensor];
                            const auto &sources =
                                previous.readers.empty() ? previous.writers : previous.readers;
                            for (int source : sources) {
                                if (source != u) {
                                    task.memory_edges.emplace_back(source, u);
                                }
                            }
                        }
                        bytes -= take;
                        credit.bytes -= take;
                        if (!credit.bytes) {
                            pool.pop_front();
                        }
                    }
                    if (bytes) {
                        throw InvalidReview{};
                    }
                    memory.resident[t] = true;
                    memory.used[b.memory] += b.size;
                }
                running[p] = {u, now + task.ops[u].duration, task.ops[u].duration, uses_ddr(task, u)};
                progress = true;
            }
        }
        if (done == n) {
            break;
        }

        project(running, now);
        double next = std::numeric_limits<double>::infinity();
        for (const auto &r : running) {
            if (r.node >= 0) {
                next = std::min(next, r.end);
            }
        }
        if (!std::isfinite(next) || next <= now) {
            throw InvalidReview{};
        }
        now = next;
    }
    distinct(task.memory_edges);

    for (auto &op : task.ops) {
        op.succs.clear();
    }
    for (const auto &edge : task.memory_edges) {
        task.ops[edge.second].preds.push_back(edge.first);
    }
    for (int u = 0; u < n; ++u) {
        distinct(task.ops[u].preds);
        for (int p : task.ops[u].preds) {
            task.ops[p].succs.push_back(u);
        }
    }
}

}
}
