#include "model.hpp"

#include <cmath>
#include <deque>
#include <list>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace npu {
namespace {

struct Buffer {
    Id id;
    int storage;
    Cost bytes;
    std::vector<int> writers, readers;
    int original = -1;
};
struct Node {
    Id id;
    int pipe, copy, bucket;
    Cost duration;
    std::vector<int> inputs, outputs, pred, succ;
    int original = -1;
};
struct Task {
    std::vector<Buffer> buffers;
    std::vector<Node> nodes;
    std::vector<std::pair<int, int>> direct;
    std::vector<int> sequence;
    std::array<std::vector<int>, kPipeCount> pipes;
    Cost spill_bytes = 0;
    std::vector<std::pair<int, Cost>> spill_by_tensor;
};

struct Link { int source_core, source, target_core, target; };

Cost checked_add(Cost a, Cost b) {
    if (a > kInfinity - b) throw std::runtime_error("performance model integer range exceeded");
    return a + b;
}

void dependencies(Task& task) {
    for (auto& b : task.buffers) { b.writers.clear(); b.readers.clear(); }
    for (int i = 0; i < static_cast<int>(task.nodes.size()); ++i) {
        auto& op = task.nodes[i];
        op.pred.clear(); op.succ.clear();
        for (int t : op.inputs) task.buffers[t].readers.push_back(i);
        for (int t : op.outputs) task.buffers[t].writers.push_back(i);
    }
    for (auto& b : task.buffers) {
        unique_sort(b.writers); unique_sort(b.readers);
        for (int w : b.writers) for (int r : b.readers)
            if (w != r) task.nodes[r].pred.push_back(w);
    }
    for (const auto& e : task.direct) if (e.first != e.second) task.nodes[e.second].pred.push_back(e.first);
    for (int i = 0; i < static_cast<int>(task.nodes.size()); ++i) {
        unique_sort(task.nodes[i].pred);
        for (int p : task.nodes[i].pred) task.nodes[p].succ.push_back(i);
    }
}

void make_sequence(Task& t) {
    dependencies(t);
    const int n = static_cast<int>(t.nodes.size());
    std::vector<int> depth(n), degree(n), queue;
    for (int i = 0; i < n; ++i) {
        degree[i] = static_cast<int>(t.nodes[i].pred.size());
        if (!degree[i]) queue.push_back(i);
    }
    for (std::size_t i = 0; i < queue.size(); ++i) {
        const int u = queue[i];
        for (int v : t.nodes[u].succ) {
            depth[v] = std::max(depth[v], depth[u] + 1);
            if (--degree[v] == 0) queue.push_back(v);
        }
    }
    if (queue.size() != t.nodes.size()) throw std::runtime_error("local dependency cycle");
    const auto less = [&t, &depth](int a, int b, int copy_kind) {
        const bool x = t.nodes[a].copy != copy_kind, y = t.nodes[b].copy != copy_kind;
        if (x != y) return x < y;
        if (depth[a] != depth[b]) return depth[a] < depth[b];
        return t.nodes[a].id > t.nodes[b].id;
    };
    std::vector<int> stack;
    for (int i = 0; i < n; ++i) if (t.nodes[i].succ.empty()) stack.push_back(i);
    std::sort(stack.begin(), stack.end(), [&](int a, int b) { return less(a, b, 2); });
    std::vector<char> done(n);
    while (!stack.empty()) {
        const int u = stack.back();
        if (done[u]) { stack.pop_back(); continue; }
        std::vector<int> pred;
        for (int p : t.nodes[u].pred) if (!done[p]) pred.push_back(p);
        if (pred.empty()) { done[u] = 1; t.sequence.push_back(u); stack.pop_back(); }
        else {
            std::sort(pred.begin(), pred.end(), [&](int a, int b) { return less(a, b, 1); });
            stack.insert(stack.end(), pred.begin(), pred.end());
        }
    }
    std::stable_sort(t.sequence.begin(), t.sequence.end(), [&](int a, int b) {
        return t.nodes[a].bucket < t.nodes[b].bucket;
    });
    std::vector<int> position(n);
    for (int i = 0; i < n; ++i) position[t.sequence[i]] = i;
    for (int i = 0; i < n; ++i) for (int p : t.nodes[i].pred)
        if (position[p] >= position[i]) throw std::runtime_error("subgraph order is not topological");
}

std::vector<Task> build_tasks(const Graph& g, const Plan& plan, std::vector<Link>& links) {
    const int cores = static_cast<int>(plan.orders.size());
    std::vector<Task> tasks(cores);
    std::vector<int> owner(g.ops.size(), -1), rank(g.ops.size(), -1), local(g.ops.size(), -1);
    for (int c = 0; c < cores; ++c)
        for (int i = 0; i < static_cast<int>(plan.orders[c].size()); ++i) {
            const int op = plan.orders[c][i]; owner[op] = c; rank[op] = i;
        }

    // Preserve original numeric ordering even when IDs occupy the end of uint64.
    const bool remap = g.max_id > UINT64_MAX - 100000000ULL;
    std::vector<Id> ordered_ids;
    if (remap) {
        for (const auto& op : g.ops) ordered_ids.push_back(op.id);
        for (const auto& t : g.tensors) ordered_ids.push_back(t.id);
        std::sort(ordered_ids.begin(), ordered_ids.end());
    }
    const auto model_id = [&](Id id) -> Id {
        if (!remap) return id;
        return static_cast<Id>(std::lower_bound(ordered_ids.begin(), ordered_ids.end(), id) - ordered_ids.begin());
    };
    Id next = remap ? static_cast<Id>(ordered_ids.size()) : g.max_id + 1;
    for (int op : g.work) {
        if (owner[op] < 0) throw std::runtime_error("missing operation");
        auto& t = tasks[owner[op]];
        const auto& original = g.ops[op];
        local[op] = static_cast<int>(t.nodes.size());
        t.nodes.push_back(Node{model_id(original.id), original.pipe, 0, rank[op],
                               std::max<Cost>(1, original.cycles), {}, {}, {}, {}, op});
    }
    const auto copy = [&](int core, int tid, Cost bytes, int kind, int bucket, Id backing) {
        auto& task = tasks[core];
        const int ddr = static_cast<int>(task.buffers.size());
        task.buffers.push_back(Buffer{backing, -1, bytes, {}, {}});
        const int index = static_cast<int>(task.nodes.size());
        Node op{next++, kind == 1 ? 0 : 1, kind, bucket, transfer_cycles(bytes), {}, {}, {}, {}};
        op.inputs.push_back(kind == 1 ? ddr : tid);
        op.outputs.push_back(kind == 1 ? tid : ddr);
        task.nodes.push_back(std::move(op));
        return index;
    };
    for (int original = 0; original < static_cast<int>(g.tensors.size()); ++original) {
        const auto& source = g.tensors[original];
        std::vector<int> producers, consumers, touched, source_cores, target_cores;
        bool final_output = false;
        for (int p : source.producers) if (!g.ops[p].copy) {
            producers.push_back(p); touched.push_back(owner[p]); source_cores.push_back(owner[p]);
        }
        for (int u : source.consumers) {
            if (!g.ops[u].copy) { consumers.push_back(u); touched.push_back(owner[u]); target_cores.push_back(owner[u]); }
            if (g.ops[u].copy == 2) final_output = true;
        }
        unique_sort(touched); unique_sort(source_cores); unique_sort(target_cores);
        std::vector<int> tid(cores, -1), first(cores, INT32_MAX), last(cores, -1);
        for (int core : touched) {
            auto& task = tasks[core];
            tid[core] = static_cast<int>(task.buffers.size());
            task.buffers.push_back(Buffer{model_id(source.id), source.storage == -1 ? 1 : source.storage,
                                          source.bytes, {}, {}, original});
            for (int p : producers) if (owner[p] == core) {
                task.nodes[local[p]].outputs.push_back(tid[core]);
                last[core] = std::max(last[core], rank[p]);
            }
            for (int u : consumers) if (owner[u] == core) {
                task.nodes[local[u]].inputs.push_back(tid[core]);
                first[core] = std::min(first[core], rank[u]);
            }
        }
        if (producers.empty()) for (int c : target_cores) {
            const Id backing = next++;
            copy(c, tid[c], source.bytes, 1, first[c], backing);
        }
        if (final_output || consumers.empty()) for (int c : source_cores) {
            const Id backing = next++;
            copy(c, tid[c], source.bytes, 2, last[c], backing);
        }
        for (int s : source_cores) for (int d : target_cores) if (s != d) {
            const Id backing = next++;
            const int out = copy(s, tid[s], source.bytes, 2, last[s], backing);
            const int in = copy(d, tid[d], source.bytes, 1, first[d], backing);
            links.push_back(Link{s, out, d, in});
        }
    }
    for (std::size_t i = 0; i < g.direct.size(); ++i) {
        const auto& e = g.direct[i];
        if (g.ops[e.first].copy || g.ops[e.second].copy) continue;
        const int s = owner[e.first], d = owner[e.second];
        if (s == d) { tasks[s].direct.emplace_back(local[e.first], local[e.second]); continue; }
        const Cost bytes = g.direct_bytes[i];
        const Id id = next++, backing = next++;
        const int st = static_cast<int>(tasks[s].buffers.size()), dt = static_cast<int>(tasks[d].buffers.size());
        tasks[s].buffers.push_back(Buffer{id, 1, bytes, {}, {}});
        tasks[d].buffers.push_back(Buffer{id, 1, bytes, {}, {}});
        tasks[s].nodes[local[e.first]].outputs.push_back(st);
        tasks[d].nodes[local[e.second]].inputs.push_back(dt);
        const int out = copy(s, st, bytes, 2, rank[e.first], backing);
        const int in = copy(d, dt, bytes, 1, rank[e.second], backing);
        links.push_back(Link{s, out, d, in});
    }
    for (auto& t : tasks) make_sequence(t);
    return tasks;
}

struct Spill { int tensor, before, after, to = -1; };

void expand_spills(Task& task) {
    const int n = static_cast<int>(task.sequence.size());
    const int nt = static_cast<int>(task.buffers.size());
    if (!n) return;
    std::vector<std::vector<int>> uses(nt), writers(nt);
    std::vector<int> step(task.nodes.size());
    for (int i = 0; i < n; ++i) {
        const int op = task.sequence[i]; step[op] = i;
        auto involved = task.nodes[op].inputs;
        involved.insert(involved.end(), task.nodes[op].outputs.begin(), task.nodes[op].outputs.end());
        unique_sort(involved);
        for (int t : involved) if (task.buffers[t].storage >= 0) uses[t].push_back(i);
        for (int t : task.nodes[op].outputs) if (task.buffers[t].storage >= 0) writers[t].push_back(i);
    }
    std::vector<std::vector<std::pair<int, int>>> at(n), reload(n);
    for (int t = 0; t < nt; ++t)
        for (int i = 0; i < static_cast<int>(uses[t].size()); ++i) at[uses[t][i]].emplace_back(t, i);
    std::array<std::list<int>, 2> active;
    std::vector<std::list<int>::iterator> iterator(nt);
    std::vector<char> resident(nt);
    std::vector<int> next_use(nt, INT32_MAX), consumed(nt), current(nt, -1);
    std::array<Cost, 2> used{{0, 0}};
    std::vector<Spill> spills;
    const auto allocate = [&](int t) {
        const int pos = task.buffers[t].storage;
        active[pos].push_back(t); iterator[t] = std::prev(active[pos].end());
        resident[t] = 1; used[pos] += task.buffers[t].bytes;
    };
    const auto release = [&](int t) {
        const int pos = task.buffers[t].storage;
        active[pos].erase(iterator[t]); resident[t] = 0; used[pos] -= task.buffers[t].bytes;
    };
    for (int i = 0; i < n; ++i) {
        for (const auto& event : at[i]) current[event.first] = i;
        for (const auto& event : reload[i]) {
            const int t = event.first, index = event.second;
            if (!resident[t]) {
                allocate(t); consumed[t] = index;
                next_use[t] = index + 1 < static_cast<int>(uses[t].size()) ? uses[t][index + 1] : INT32_MAX;
            }
        }
        for (const auto& event : at[i]) {
            const int t = event.first, index = event.second;
            if (index == 0) {
                allocate(t); consumed[t] = 1;
                next_use[t] = uses[t].size() > 1 ? uses[t][1] : INT32_MAX;
            } else if (index + 1 < static_cast<int>(uses[t].size()) && resident[t]) {
                next_use[t] = uses[t][index + 1]; ++consumed[t];
            }
        }
        for (int p = 0; p < 2; ++p) while (used[p] > kSpillThreshold[p]) {
            int victim = -1;
            for (int t : active[p])
                if (current[t] != i && next_use[t] != INT32_MAX &&
                    (victim == -1 || next_use[t] > next_use[victim])) victim = t;
            if (victim == -1) {
                // Current operands cannot be evicted. Their transient peak may
                // exceed the soft threshold, but never the physical capacity.
                if (used[p] <= kCapacity[p]) break;
                throw std::runtime_error("an operation exceeds on-chip capacity");
            }
            const int previous = uses[victim][std::max(0, consumed[victim] - 1)];
            spills.push_back(Spill{victim, next_use[victim], previous, -1});
            reload[next_use[victim]].emplace_back(victim, consumed[victim]);
            release(victim);
        }
        for (const auto& event : at[i])
            if (event.second + 1 == static_cast<int>(uses[event.first].size()) && resident[event.first]) release(event.first);
    }

    Id next_id = 0;
    for (const auto& t : task.buffers) next_id = std::max(next_id, t.id + 1);
    for (const auto& op : task.nodes) next_id = std::max(next_id, op.id + 1);
    std::vector<int> backing(nt, -1), physical(nt), synced_step(nt, -1);
    std::iota(physical.begin(), physical.end(), 0);
    for (const auto& op : task.nodes) if (op.copy == 1) {
        int ddr = -1, count = 0;
        for (int t : op.inputs) if (task.buffers[t].storage == -1) { ddr = t; ++count; }
        if (count == 1) for (int t : op.outputs) if (task.buffers[t].storage >= 0) {
            if (backing[t] == -1) backing[t] = ddr;
            else if (backing[t] != ddr) backing[t] = -2; // Conflicting COPY_IN backings.
        }
    }
    for (int t = 0; t < nt; ++t) {
        if (backing[t] >= 0 && !uses[t].empty()) synced_step[t] = uses[t].front();
        else backing[t] = -1;
    }
    const int original_nodes = static_cast<int>(task.nodes.size());
    std::vector<std::vector<int>> before(n), after(n), by_tensor(nt);
    for (int i = 0; i < static_cast<int>(spills.size()); ++i) {
        auto& sp = spills[i];
        const int logical = sp.tensor, old = physical[logical];
        const Cost bytes = task.buffers[logical].bytes;
        const int pos = task.buffers[logical].storage;
        const auto next_writer = std::upper_bound(writers[logical].begin(), writers[logical].end(), synced_step[logical]);
        const bool dirty = next_writer != writers[logical].end() && *next_writer <= sp.after;
        const bool write_back = backing[logical] == -1 || dirty;
        const Id out_id = write_back ? next_id++ : 0;
        const Id in_id = next_id++;
        if (write_back) {
            // Each dirty writeback gets a new DDR version; older reloads must
            // keep depending on the data saved before the later writers.
            synced_step[logical] = sp.after;
            backing[logical] = static_cast<int>(task.buffers.size());
            task.buffers.push_back(Buffer{next_id++, -1, bytes, {}, {}});
        }
        const int fresh = static_cast<int>(task.buffers.size());
        task.buffers.push_back(Buffer{next_id++, pos, bytes, {}, {}});
        physical[logical] = fresh; sp.to = fresh; by_tensor[logical].push_back(i);
        if (write_back) {
            after[sp.after].push_back(static_cast<int>(task.nodes.size()));
            task.nodes.push_back(Node{out_id, 1, 2, 0, transfer_cycles(bytes), {old}, {backing[logical]}, {}, {}});
            task.spill_bytes = checked_add(task.spill_bytes, bytes);
        }
        before[sp.before].push_back(static_cast<int>(task.nodes.size()));
        task.nodes.push_back(Node{in_id, 0, 1, 0, transfer_cycles(bytes), {backing[logical]}, {fresh}, {}, {}});
        task.spill_bytes = checked_add(task.spill_bytes, bytes);
        if (task.buffers[logical].original >= 0)
            task.spill_by_tensor.emplace_back(task.buffers[logical].original, bytes * (write_back ? 2 : 1));
    }
    for (int op = 0; op < original_nodes; ++op) {
        const auto rewire = [&](std::vector<int>& operands) {
            for (int& t : operands) {
                const auto& records = by_tensor[t];
                const auto it = std::upper_bound(records.begin(), records.end(), step[op], [&](int s, int r) {
                    return s < spills[r].before;
                });
                if (it != records.begin()) t = spills[*std::prev(it)].to;
            }
        };
        rewire(task.nodes[op].inputs);
        rewire(task.nodes[op].outputs);
    }
    std::vector<int> sequence;
    sequence.reserve(task.nodes.size());
    for (int i = 0; i < n; ++i) {
        if (i) sequence.insert(sequence.end(), after[i - 1].begin(), after[i - 1].end());
        sequence.insert(sequence.end(), before[i].begin(), before[i].end());
        sequence.push_back(task.sequence[i]);
    }
    sequence.insert(sequence.end(), after.back().begin(), after.back().end());
    task.sequence = std::move(sequence);
    dependencies(task);
    std::vector<int> position(task.nodes.size());
    for (int i = 0; i < static_cast<int>(task.sequence.size()); ++i) position[task.sequence[i]] = i;
    for (int op : task.sequence) for (int pred : task.nodes[op].pred)
        if (position[pred] >= position[op]) throw std::runtime_error("spill expansion violates dependency order");
}

// Reference-compatible integer-set visitation for operand retirement. This
// affects the order of reusable memory credits, not the chosen task order.
std::vector<int> operand_set_order(const Task& task, const std::vector<int>& values) {
    if (values.empty()) return {};
    if (values.size() == 1) return {values.front()};
    std::vector<int> table(8, -1);
    std::size_t used = 0;
    const auto insert = [&](std::vector<int>& slots, int value) {
        const std::uint64_t hash = task.buffers[value].id % 2305843009213693951ULL;
        const std::size_t mask = slots.size() - 1;
        std::uint64_t perturb = hash;
        std::size_t i = static_cast<std::size_t>(hash) & mask;
        for (;;) {
            const std::size_t count = i + 9 <= mask ? 9 : 0;
            for (std::size_t j = 0; j <= count; ++j) {
                if (slots[i + j] == value) return false;
                if (slots[i + j] == -1) { slots[i + j] = value; return true; }
            }
            perturb >>= 5;
            i = (i * 5 + 1 + static_cast<std::size_t>(perturb)) & mask;
        }
    };
    for (int value : values) {
        if (!insert(table, value)) continue;
        ++used;
        if (used * 5 >= (table.size() - 1) * 3) {
            std::size_t size = 8;
            while (size <= used * (used > 50000 ? 2 : 4)) size *= 2;
            std::vector<int> larger(size, -1);
            for (int old : table) if (old >= 0) insert(larger, old);
            table = std::move(larger);
        }
    }
    std::vector<int> result;
    for (int value : table) if (value >= 0) result.push_back(value);
    return result;
}

struct Running { int op = -1; Cost end = 0; double work = -1.0; };
struct Bus {
    std::vector<Running> running;
    double last = 0;
    explicit Bus(int slots) : running(slots) {}
    void advance(Cost now) {
        double elapsed = static_cast<double>(now) - last;
        while (elapsed > 1e-9) {
            int count = 0;
            double first = static_cast<double>(kInfinity);
            for (const auto& r : running) if (r.op >= 0 && r.work > 1e-9) { ++count; first = std::min(first, r.work); }
            if (!count) break;
            if (first * count >= elapsed - 1e-9) {
                for (auto& r : running) if (r.op >= 0 && r.work > 1e-9) r.work = std::max(0.0, r.work - elapsed / count);
                break;
            }
            for (auto& r : running) if (r.op >= 0 && r.work > 1e-9) r.work = std::max(0.0, r.work - first);
            elapsed -= first * count;
        }
        last = static_cast<double>(now);
    }
    void project(Cost now) {
        std::vector<std::pair<double, int>> pending;
        for (int i = 0; i < static_cast<int>(running.size()); ++i)
            if (running[i].op >= 0 && running[i].work >= 0) pending.emplace_back(running[i].work, i);
        std::sort(pending.begin(), pending.end());
        double previous = 0, cursor = static_cast<double>(now);
        int count = static_cast<int>(pending.size());
        for (int i = 0; i < static_cast<int>(pending.size()); ) {
            const double work = pending[i].first;
            cursor += (work - previous) * count;
            int j = i;
            while (j < static_cast<int>(pending.size()) && std::abs(pending[j].first - work) <= 1e-9) {
                if (cursor >= static_cast<double>(kInfinity)) throw std::runtime_error("timeline overflow");
                running[pending[j].second].end = static_cast<Cost>(std::ceil(cursor - 1e-9));
                ++j;
            }
            count -= j - i; previous = work; i = j;
        }
    }
    Cost next_time() const {
        Cost next = kInfinity;
        for (const auto& r : running) if (r.op >= 0) next = std::min(next, r.end);
        return next;
    }
};

void prepare_execution(Task& task) {
    const int n = static_cast<int>(task.nodes.size());
    if (!n) return;
    std::vector<int> degree(n), remaining(task.buffers.size()), allocation, allocation_rank(n, -1);
    std::vector<std::vector<int>> inputs(n), outputs(n);
    std::vector<char> resident(task.buffers.size());
    std::array<Cost, 2> used{{0, 0}};
    struct Credit { Cost bytes; int tensor; };
    std::array<std::deque<Credit>, 2> credits;
    for (int t = 0; t < static_cast<int>(task.buffers.size()); ++t) {
        const auto& b = task.buffers[t];
        remaining[t] = static_cast<int>(b.readers.size());
        if (b.storage >= 0 && b.writers.empty() && !b.readers.empty()) {
            resident[t] = 1; used[b.storage] += b.bytes;
        }
    }
    for (int p = 0; p < 2; ++p) {
        if (used[p] > kCapacity[p]) throw std::runtime_error("initial memory overflow");
        if (used[p] < kCapacity[p]) credits[p].push_back(Credit{kCapacity[p] - used[p], -1});
    }
    for (int op : task.sequence) {
        task.pipes[task.nodes[op].pipe].push_back(op);
        degree[op] = static_cast<int>(task.nodes[op].pred.size());
        inputs[op] = operand_set_order(task, task.nodes[op].inputs);
        outputs[op] = operand_set_order(task, task.nodes[op].outputs);
        for (int t : task.nodes[op].outputs) if (task.buffers[t].storage >= 0) {
            allocation_rank[op] = static_cast<int>(allocation.size()); allocation.push_back(op); break;
        }
    }
    const auto release = [&](int t) {
        if (!resident[t]) return;
        const auto& b = task.buffers[t];
        resident[t] = 0; used[b.storage] -= b.bytes;
        credits[b.storage].push_back(Credit{b.bytes, t});
    };
    std::array<std::size_t, kPipeCount> cursor{};
    std::size_t alloc_cursor = 0;
    int completed = 0;
    Cost now = 0;
    Bus bus(kPipeCount);
    while (completed < n) {
        bus.advance(now);
        bool retired_ddr = false;
        for (int pipe = 0; pipe < kPipeCount; ++pipe) {
            auto& running = bus.running[pipe];
            if (running.op == -1 || running.end > now) continue;
            const int op = running.op;
            for (int t : inputs[op]) if (task.buffers[t].storage >= 0 && --remaining[t] == 0) release(t);
            for (int t : outputs[op]) if (task.buffers[t].storage >= 0 && remaining[t] == 0) release(t);
            for (int v : task.nodes[op].succ) --degree[v];
            retired_ddr = retired_ddr || running.work >= 0;
            running = Running{}; ++cursor[pipe]; ++completed;
        }
        if (retired_ddr) bus.project(now);
        bool issued;
        do {
            issued = false;
            for (int pipe = 0; pipe < kPipeCount; ++pipe) {
                if (bus.running[pipe].op >= 0 || cursor[pipe] == task.pipes[pipe].size()) continue;
                const int op = task.pipes[pipe][cursor[pipe]];
                if (degree[op]) continue;
                if (allocation_rank[op] >= 0 &&
                    (alloc_cursor == allocation.size() || allocation[alloc_cursor] != op)) continue;
                auto allocated = task.nodes[op].outputs;
                std::sort(allocated.begin(), allocated.end(), [&](int a, int b) { return task.buffers[a].id < task.buffers[b].id; });
                allocated.erase(std::unique(allocated.begin(), allocated.end()), allocated.end());
                std::array<Cost, 2> requested{{0, 0}};
                for (int t : allocated) if (task.buffers[t].storage >= 0 && !resident[t]) requested[task.buffers[t].storage] += task.buffers[t].bytes;
                if (requested[0] + used[0] > kCapacity[0] || requested[1] + used[1] > kCapacity[1]) continue;
                for (int t : allocated) {
                    const auto& b = task.buffers[t];
                    if (b.storage < 0 || resident[t]) continue;
                    Cost amount = b.bytes;
                    auto& free = credits[b.storage];
                    while (amount > 0) {
                        if (free.empty()) throw std::runtime_error("memory credits exhausted");
                        auto& credit = free.front();
                        const Cost take = std::min(amount, credit.bytes);
                        if (credit.tensor >= 0 && take > 0) {
                            const auto& previous = task.buffers[credit.tensor];
                            const auto& sources = previous.readers.empty() ? previous.writers : previous.readers;
                            for (int src : sources) if (src != op) task.direct.emplace_back(src, op);
                        }
                        amount -= take; credit.bytes -= take;
                        if (!credit.bytes) free.pop_front();
                    }
                    resident[t] = 1; used[b.storage] += b.bytes;
                }
                if (allocation_rank[op] >= 0) ++alloc_cursor;
                auto& running = bus.running[pipe];
                running.op = op; running.end = checked_add(now, task.nodes[op].duration);
                if (task.nodes[op].copy) {
                    bus.advance(now); running.work = static_cast<double>(task.nodes[op].duration); bus.project(now);
                }
                issued = true;
            }
        } while (issued);
        if (completed == n) break;
        const Cost next = bus.next_time();
        if (next == kInfinity || next <= now) throw std::runtime_error("local memory/pipe deadlock");
        now = next;
    }
    if (used[0] || used[1]) throw std::runtime_error("unclosed memory lifetime");
    for (int t = 0; t < static_cast<int>(task.buffers.size()); ++t)
        if (task.buffers[t].storage >= 0 && remaining[t]) throw std::runtime_error("unclosed tensor use");
    dependencies(task);
}

Cost simulate_global(const std::vector<Task>& tasks, const std::vector<Link>& links, Evaluation& result) {
    struct Event {
        int pipe = 0, original = -1;
        Cost duration = 0, release = 0;
        bool ddr = false;
        std::vector<int> next, cross;
        int degree = 0;
    };
    std::vector<int> offsets(tasks.size() + 1);
    for (std::size_t c = 0; c < tasks.size(); ++c) offsets[c + 1] = offsets[c] + static_cast<int>(tasks[c].nodes.size());
    std::vector<Event> events(static_cast<std::size_t>(offsets.back()));
    std::vector<std::vector<int>> orders(tasks.size() * kPipeCount);
    for (int c = 0; c < static_cast<int>(tasks.size()); ++c) {
        const auto& task = tasks[c];
        for (int i = 0; i < static_cast<int>(task.nodes.size()); ++i) {
            auto& event = events[offsets[c] + i];
            const auto& op = task.nodes[i];
            event.pipe = c * kPipeCount + op.pipe; event.duration = op.duration; event.ddr = op.copy != 0;
            event.original = op.original;
            event.degree = static_cast<int>(op.pred.size());
            for (int v : op.succ) event.next.push_back(offsets[c] + v);
        }
        for (int pipe = 0; pipe < kPipeCount; ++pipe)
            for (int op : task.pipes[pipe]) orders[c * kPipeCount + pipe].push_back(offsets[c] + op);
    }
    for (const auto& link : links) {
        const int src = offsets[link.source_core] + link.source, dst = offsets[link.target_core] + link.target;
        events[src].cross.push_back(dst); ++events[dst].degree;
    }
    result.core_end.assign(tasks.size(), 0);
    std::vector<std::size_t> cursor(orders.size());
    Bus bus(static_cast<int>(orders.size()));
    int completed = 0;
    Cost now = 0;
    while (completed < static_cast<int>(events.size())) {
        bus.advance(now);
        bool retired_ddr = false;
        for (int p = 0; p < static_cast<int>(orders.size()); ++p) {
            auto& r = bus.running[p];
            if (r.op < 0 || r.end > now) continue;
            for (int v : events[r.op].next) --events[v].degree;
            for (int v : events[r.op].cross) {
                --events[v].degree;
                events[v].release = std::max(events[v].release, checked_add(now, kCrossCoreDelay));
            }
            const int original = events[r.op].original;
            if (original >= 0) result.op_end[original] = now;
            result.core_end[p / kPipeCount] = now;
            retired_ddr = retired_ddr || r.work >= 0;
            r = Running{}; ++completed; ++cursor[p];
        }
        if (retired_ddr) bus.project(now);
        for (int p = 0; p < static_cast<int>(orders.size()); ++p) {
            if (bus.running[p].op >= 0 || cursor[p] == orders[p].size()) continue;
            const int op = orders[p][cursor[p]];
            const auto& event = events[op];
            if (event.degree || event.release > now) continue;
            auto& r = bus.running[p]; r.op = op; r.end = checked_add(now, event.duration);
            if (event.original >= 0) result.op_start[event.original] = now;
            if (event.ddr) { bus.advance(now); r.work = static_cast<double>(event.duration); bus.project(now); }
        }
        if (completed == static_cast<int>(events.size())) break;
        Cost next = bus.next_time();
        for (int p = 0; p < static_cast<int>(orders.size()); ++p) {
            if (bus.running[p].op >= 0 || cursor[p] == orders[p].size()) continue;
            const auto& event = events[orders[p][cursor[p]]];
            if (!event.degree && event.release > now) next = std::min(next, event.release);
        }
        if (next == kInfinity || next <= now) throw std::runtime_error("global pipe/dependency deadlock");
        now = next;
    }
    return now;
}

Cost necessary_io(const Graph& g, const std::vector<int>* owner) {
    Cost service = 0;
    for (const auto& t : g.tensors) {
        std::vector<int> producers, consumers;
        bool output = false;
        for (int p : t.producers) if (!g.ops[p].copy) producers.push_back(owner ? (*owner)[p] : 0);
        for (int u : t.consumers) {
            if (!g.ops[u].copy) consumers.push_back(owner ? (*owner)[u] : 0);
            if (g.ops[u].copy == 2) output = true;
        }
        unique_sort(producers); unique_sort(consumers);
        int copies = 0;
        if (producers.empty()) copies += static_cast<int>(consumers.size());
        else if (output || consumers.empty()) copies += static_cast<int>(producers.size());
        for (int p : producers) for (int c : consumers) if (p != c) copies += 2;
        for (int i = 0; i < copies; ++i) service = checked_add(service, transfer_cycles(t.bytes));
    }
    if (owner) for (std::size_t i = 0; i < g.direct.size(); ++i) {
        const auto& e = g.direct[i];
        if (!g.ops[e.first].copy && !g.ops[e.second].copy && (*owner)[e.first] != (*owner)[e.second]) {
            service = checked_add(service, transfer_cycles(g.direct_bytes[i]));
            service = checked_add(service, transfer_cycles(g.direct_bytes[i]));
        }
    }
    return service;
}
} // namespace

// 所有分核方案的执行时间下界，取以下三项的最大值。
Cost universal_lower_bound(const Graph& g, int cores) {
    std::array<Cost, kPipeCount> work{};
    // 必要输入读取和结果写回占用的 DDR 时间。
    Cost bound = necessary_io(g, nullptr);
    for (int op : g.work) {
        const Cost time = std::max<Cost>(1, g.ops[op].cycles);
        work[g.ops[op].pipe] = checked_add(work[g.ops[op].pipe], time);
        bound = std::max(bound, time); // 至少覆盖最长单个操作的耗时。
    }
    // 各 Pipe 的总计算量均分到所有核，耗时向上取整。
    for (Cost w : work) bound = std::max(bound, w / cores + (w % cores != 0));
    return bound;
}

Cost partition_lower_bound(const Graph& g, const Plan& plan) {
    std::vector<int> owner(g.ops.size(), -1);
    Cost bound = 0;
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c) {
        std::array<Cost, kPipeCount> work{};
        for (int op : plan.orders[c]) {
            owner[op] = c;
            work[g.ops[op].pipe] = checked_add(work[g.ops[op].pipe], std::max<Cost>(1, g.ops[op].cycles));
        }
        for (Cost w : work) bound = std::max(bound, w);
    }
    return std::max(bound, necessary_io(g, &owner));
}

Cost partition_ddr_cycles(const Graph& g, const Plan& plan) {
    std::vector<int> owner(g.ops.size(), -1);
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c)
        for (int op : plan.orders[c]) owner[op] = c;
    return necessary_io(g, &owner);
}

Evaluation evaluate(const Graph& g, const Plan& plan, Cost incumbent) {
    Evaluation result;
    try {
        if (!valid_plan(g, plan)) { result.error = "invalid plan"; return result; }
        if (partition_lower_bound(g, plan) > incumbent) { result.error = "partition lower bound"; return result; }
        std::vector<Link> links;
        auto tasks = build_tasks(g, plan, links);
        result.tensor_spill_bytes.assign(g.tensors.size(), 0);
        Cost pipe_bound = 0;
        for (auto& t : tasks) {
            expand_spills(t);
            result.spill_bytes = checked_add(result.spill_bytes, t.spill_bytes);
            for (const auto& item : t.spill_by_tensor)
                result.tensor_spill_bytes[item.first] = checked_add(result.tensor_spill_bytes[item.first], item.second);
            std::array<Cost, kPipeCount> work{};
            for (const auto& op : t.nodes) {
                work[op.pipe] = checked_add(work[op.pipe], op.duration);
                if (op.copy) {
                    result.ddr_cycles = checked_add(result.ddr_cycles, op.duration);
                    const auto& data = op.copy == 1 ? op.outputs : op.inputs;
                    for (int tid : data) result.bytes = checked_add(result.bytes, t.buffers[tid].bytes);
                }
            }
            for (Cost w : work) pipe_bound = std::max(pipe_bound, w);
        }
        if (std::max(pipe_bound, result.ddr_cycles) > incumbent) {
            result.error = "expanded lower bound"; return result;
        }
        for (auto& t : tasks) prepare_execution(t);
        result.op_start.assign(g.ops.size(), 0); result.op_end.assign(g.ops.size(), 0);
        result.makespan = simulate_global(tasks, links, result);
        result.valid = true;
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}
} // namespace npu
