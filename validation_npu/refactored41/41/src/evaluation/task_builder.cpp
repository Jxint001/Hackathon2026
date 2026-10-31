#include "evaluation/model.hpp"
namespace npu {
namespace evaluation_detail {
int buffer(Task &task, Key id, Id bytes, int memory) {
    task.buffers.emplace_back(id, bytes, memory, task.memory);
    return static_cast<int>(task.buffers.size()) - 1;
}
int copy(Task &task, Key id, int kind, int input, int output, int group) {
    Instruction op(task.memory);
    op.id = id;
    op.copy = kind;
    op.pipe = kind == 1 ? 2 : 3;
    op.group = group;
    op.inputs = {input};
    op.outputs = {output};
    op.duration = std::max(1.0, std::ceil(task.buffers[kind == 1 ? output : input].size / DDR_BANDWIDTH));
    task.ops.push_back(std::move(op));
    return static_cast<int>(task.ops.size()) - 1;
}
void build_views(Task &task) {
    for (auto &t : task.buffers) {
        t.writers.clear();
        t.readers.clear();
    }
    for (int u = 0; u < static_cast<int>(task.ops.size()); ++u) {
        auto &op = task.ops[u];
        distinct(op.inputs);
        distinct(op.outputs);
        op.preds.clear();
        op.succs.clear();
        for (int t : op.inputs) {
            task.buffers[t].readers.push_back(u);
        }
        for (int t : op.outputs) {
            task.buffers[t].writers.push_back(u);
        }
    }
    for (const auto &t : task.buffers) {
        for (int p : t.writers) {
            for (int u : t.readers) {
                if (p != u) {
                    task.ops[u].preds.push_back(p);
                }
            }
        }
    }
    for (const auto &edge : task.memory_edges) {
        task.ops[edge.second].preds.push_back(edge.first);
    }
    for (int u = 0; u < static_cast<int>(task.ops.size()); ++u) {
        distinct(task.ops[u].preds);
        for (int p : task.ops[u].preds) {
            task.ops[p].succs.push_back(u);
        }
    }
}
void build_tasks(const Problem &g, const std::vector<int> &mapping, const Plan &solution, int cores,
                 EvaluationWorkspace &workspace, PlanEvaluation &result) {
    const auto group_count = solution.core.size();
    if (cores < 1 || cores > 5 || mapping.size() != g.ops.size() || solution.order.size() != group_count) {
        throw InvalidReview{};
    }
    std::vector<char> seen(group_count);
    for (int u : solution.order) {
        if (u < 0 || u >= static_cast<int>(seen.size()) || seen[u] || solution.core[u] < 0 ||
            solution.core[u] >= cores) {
            throw InvalidReview{};
        }
        seen[u] = true;
    }

    auto &arena = workspace.arena;
    std::pmr::vector<Indices> group_edges(group_count, &arena);
    for (int u : g.op_topo) {
        int a = mapping[u];
        if (a < 0 || a >= static_cast<int>(group_count)) {
            throw InvalidReview{};
        }
        for (int v : g.ops[u].succs) {
            int b = mapping[v];
            if (b < 0 || b >= static_cast<int>(group_count)) {
                throw InvalidReview{};
            }
            if (a != b) {
                group_edges[a].push_back(b);
            }
        }
    }
    std::vector<int> degrees(group_count), ready;
    for (auto &edges : group_edges) {
        distinct(edges);
        for (int v : edges) {
            ++degrees[v];
        }
    }
    for (int u = 0; u < static_cast<int>(degrees.size()); ++u) {
        if (!degrees[u]) {
            ready.push_back(u);
        }
    }
    for (std::size_t i = 0; i < ready.size(); ++i) {
        for (int v : group_edges[ready[i]]) {
            if (!--degrees[v]) {
                ready.push_back(v);
            }
        }
    }
    if (ready.size() != group_count) {
        throw InvalidReview{};
    }
    for (const auto &tensor : g.tensors) {
        for (int u : tensor.producers) {
            if (solution.core[mapping[u]] != solution.core[mapping[tensor.producers.front()]]) {
                throw InvalidReview{};
            }
        }
    }

    auto &tasks = workspace.tasks;
    tasks.reserve(cores);
    for (int c = 0; c < cores; ++c) {
        tasks.emplace_back(&arena);
    }
    std::vector<int> local_op(g.ops.size(), -1);
    auto &rank = workspace.rank;
    rank.resize(group_count);
    for (int i = 0; i < static_cast<int>(solution.order.size()); ++i) {
        rank[solution.order[i]] = i;
    }
    Key next_id = 0;
    for (const auto &op : g.ops) {
        next_id = std::max(next_id, static_cast<Key>(op.id) + 1);
    }
    for (const auto &t : g.tensors) {
        next_id = std::max(next_id, static_cast<Key>(t.id) + 1);
    }
    for (int u = 0; u < static_cast<int>(g.ops.size()); ++u) {
        if (!g.ops[u].copy) {
            int group = mapping[u], core = solution.core[group];
            auto &task = tasks[core];
            Instruction op(task.memory);
            op.id = static_cast<Key>(g.ops[u].id);
            op.pipe = g.ops[u].pipe;
            op.duration = g.ops[u].cycles;
            op.group = group;
            op.inputs.reserve(g.ops[u].inputs.size());
            op.outputs.reserve(g.ops[u].outputs.size());
            local_op[u] = static_cast<int>(task.ops.size());
            task.ops.push_back(std::move(op));
        }
    }
    auto &links = workspace.links;
    for (const auto &t : g.tensors) {
        std::array<int, 5> local, first, last;
        local.fill(-1);
        first.fill(-1);
        last.fill(-1);
        int source = t.producers.empty() ? -1 : solution.core[mapping[t.producers.front()]];
        unsigned targets = 0, touched = 0;
        for (int u : t.producers) {
            int group = mapping[u], c = solution.core[group];
            touched |= 1u << c;
            if (last[c] < 0 || rank[last[c]] < rank[group]) {
                last[c] = group;
            }
        }
        for (int u : t.consumers) {
            int group = mapping[u], c = solution.core[group];
            touched |= 1u << c;
            targets |= 1u << c;
            if (first[c] < 0 || rank[first[c]] > rank[group]) {
                first[c] = group;
            }
        }
        for (int c = 0; c < cores; ++c) {
            if (touched & (1u << c)) {
                auto &task = tasks[c];
                local[c] = buffer(task, static_cast<Key>(t.id), t.size, t.memory == 1 ? 1 : 2);
                for (int u : t.producers) {
                    if (solution.core[mapping[u]] == c) {
                        task.ops[local_op[u]].outputs.push_back(local[c]);
                    }
                }
                for (int u : t.consumers) {
                    if (solution.core[mapping[u]] == c) {
                        task.ops[local_op[u]].inputs.push_back(local[c]);
                    }
                }
            }
        }
        if (source < 0) {
            for (int c = 0; c < cores; ++c) {
                if (targets & (1u << c)) {
                    int ddr = buffer(tasks[c], next_id++, t.size, 0);
                    copy(tasks[c], next_id++, 1, ddr, local[c], first[c]);
                    result.copy_bytes += t.size;
                }
            }
        }
        if (source >= 0 && t.output) {
            int ddr = buffer(tasks[source], next_id++, t.size, 0);
            copy(tasks[source], next_id++, 2, local[source], ddr, last[source]);
            result.copy_bytes += t.size;
        }
        if (source >= 0) {
            for (int c = 0; c < cores; ++c) {
                if (c != source && (targets & (1u << c))) {
                    Key ddr_id = next_id++;
                    int out_ddr = buffer(tasks[source], ddr_id, t.size, 0);
                    int out = copy(tasks[source], next_id++, 2, local[source], out_ddr, last[source]);
                    int in_ddr = buffer(tasks[c], ddr_id, t.size, 0);
                    int in = copy(tasks[c], next_id++, 1, in_ddr, local[c], first[c]);
                    links.push_back({source, out, c, in});
                    result.copy_bytes += 2.0 * t.size;
                }
            }
        }
    }
}

}
}
