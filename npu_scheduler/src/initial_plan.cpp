// 枚举实际核心数、遍历顺序及分核策略，以快速估计和完整复核选出通用初始解。
#include "initial_plan.hpp"
#include "plan_evaluator.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>

namespace npu {
namespace {
// 快速初始解模型中的张量摘要，将读写关系投影到分组级别。
struct Net {
    double bytes = 0;
    int memory = 0;
    bool output = false;
    std::vector<int> producers, consumers, touches;
};
// 排序去重，固定遍历与张量关联顺序。
template <class T> void unique(std::vector<T> &v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}
// 管理通用初始解候选生成、快速估计、完整复核和有限排序优化。
class InitialPlanBuilder {
    const Problem &g;
    const Partition &partition;
    int cores;
    std::vector<Net> nets;
    std::vector<std::vector<int>> op_tensors;
    int evaluations = 0;
    // 初始解选优使用的二元成绩。
    struct Score {
        double time, bytes;
    };

    // 用管道时间线、跨核搬运和下一次使用距离快速估计方案。
    Score estimate_cost(const std::vector<int> &assignment, const std::vector<int> &order) {
        ++evaluations;
        std::array<std::array<double, NUM_PIPES>, 5> pipe{};
        std::vector<double> finish(g.ops.size());
        std::vector<std::array<double, 5>> ready(nets.size());
        for (auto &a : ready) {
            a.fill(-1);
        }
        std::vector<int> op_order;
        op_order.reserve(g.op_topo.size());
        for (int u : order) {
            op_order.insert(op_order.end(), partition.groups[u].ops.begin(), partition.groups[u].ops.end());
        }
        double bytes = 0;
        for (std::size_t i = 0; i < nets.size(); ++i) {
            const auto &t = nets[i];
            int src = t.producers.empty() ? -1 : assignment[t.producers.front()];
            unsigned consumers = 0;
            for (int x : t.consumers) {
                consumers |= 1u << assignment[x];
            }
            for (int c = 0; c < cores; ++c) {
                if (consumers & (1u << c)) {
                    if (src < 0) {
                        bytes += t.bytes;
                    } else if (src != c) {
                        bytes += 2 * t.bytes;
                    }
                }
            }
            if (t.output && src >= 0) {
                bytes += t.bytes;
            }
        }
        for (int u : op_order) {
            int c = assignment[partition.op_group[u]];
            const auto &a = g.ops[u];
            double begin = 0;
            for (int p : a.preds) {
                begin = std::max(begin,
                                 finish[p] + (assignment[partition.op_group[p]] != c ? CROSS_CORE_DELAY : 0));
            }
            for (int ti : a.inputs) {
                const auto &t = nets[ti];
                int src = t.producers.empty() ? -1 : assignment[t.producers.front()];
                if (src == c) {
                    continue;
                }
                auto &when = ready[ti][c];
                if (when < 0) {
                    double at = 0;
                    for (int p : g.tensors[ti].producers) {
                        at = std::max(at, finish[p]);
                    }
                    double transfer = std::max(1.0, std::ceil(t.bytes / DDR_BANDWIDTH));
                    if (src >= 0) {
                        at = std::max(at, pipe[src][3]) + transfer;
                        pipe[src][3] = at;
                        at += CROSS_CORE_DELAY;
                    }
                    when = std::max(at, pipe[c][2]) + transfer;
                    pipe[c][2] = when;
                }
                begin = std::max(begin, when);
            }
            double end = std::max(begin, pipe[c][a.pipe]) + a.cycles;
            pipe[c][a.pipe] = end;
            finish[u] = end;
        }

        const int infinity = static_cast<int>(op_order.size());
        std::vector<std::array<int, 5>> next(nets.size());
        for (auto &x : next) {
            x.fill(infinity);
        }
        std::vector<std::vector<int>> following(op_order.size());
        for (int i = infinity - 1; i >= 0; --i) {
            int u = op_order[i], c = assignment[partition.op_group[u]];
            for (int t : op_tensors[u]) {
                following[i].push_back(next[t][c]);
                next[t][c] = i;
            }
        }
        // 快速容量回放中，一个张量在一个核心上的驻留状态。
        struct Residency {
            bool resident = false, dirty = false, seen = false, pinned = false;
            int next = 0;
        };
        std::vector<std::array<Residency, 5>> state(nets.size());
        // 按下一次使用位置排序的大根堆，用于挑选可驱逐张量。
        using Heap = std::priority_queue<std::pair<int, int>>;
        std::array<std::array<Heap, 2>, 5> victims;
        std::array<std::array<double, 2>, 5> used{};
        double spill = 0;
        for (int i = 0; i < infinity; ++i) {
            int u = op_order[i], c = assignment[partition.op_group[u]];
            std::array<double, 2> missing{};
            for (int t : op_tensors[u]) {
                auto &s = state[t][c];
                s.pinned = true;
                if (!s.resident) {
                    missing[nets[t].memory] += nets[t].bytes;
                }
            }
            for (int m = 0; m < 2; ++m) {
                double limit = 0.9 * (m == 0 ? L1_CAPACITY : UB_CAPACITY);
                auto &heap = victims[c][m];
                while (used[c][m] + missing[m] > limit && !heap.empty()) {
                    auto item = heap.top();
                    heap.pop();
                    int t = item.second;
                    auto &s = state[t][c];
                    if (!s.resident || s.next != item.first || s.pinned) {
                        continue;
                    }
                    if (s.dirty) {
                        spill += nets[t].bytes;
                        s.dirty = false;
                    }
                    s.resident = false;
                    used[c][m] -= nets[t].bytes;
                }
            }
            for (std::size_t j = 0; j < op_tensors[u].size(); ++j) {
                int t = op_tensors[u][j];
                auto &s = state[t][c];
                const auto &net = nets[t];
                if (!s.resident) {
                    if (s.seen) {
                        spill += net.bytes;
                    }
                    s.resident = true;
                    used[c][net.memory] += net.bytes;
                }
                s.seen = true;
                s.next = following[i][j];
            }
            for (int t : g.ops[u].outputs) {
                state[t][c].dirty = true;
            }
            for (int t : op_tensors[u]) {
                auto &s = state[t][c];
                s.pinned = false;
                if (s.next == infinity) {
                    s.resident = false;
                    used[c][nets[t].memory] -= nets[t].bytes;
                } else {
                    victims[c][nets[t].memory].emplace(s.next, t);
                }
            }
        }
        double makespan = 0;
        for (int c = 0; c < cores; ++c) {
            for (double x : pipe[c]) {
                makespan = std::max(makespan, x);
            }
        }
        bytes += spill;

        return {std::max(makespan, bytes / DDR_BANDWIDTH) + 0.20 * spill / DDR_BANDWIDTH, bytes};
    }
    // 生成确定性的分组遍历顺序。
    std::vector<int> order_for(int mode) const {
        int n = static_cast<int>(partition.groups.size());
        if (mode == 0) {
            return partition.topo;
        }
        if (mode == 1) {
            std::vector<int> starts(n);
            std::iota(starts.begin(), starts.end(), 0);
            // 按关键路径长度降序、索引升序比较 DFS 起点。
            std::sort(starts.begin(), starts.end(), [&](int a, int b) {
                if (partition.groups[a].rank != partition.groups[b].rank) {
                    return partition.groups[a].rank > partition.groups[b].rank;
                }
                return a < b;
            });
            std::vector<char> seen(n);
            std::vector<int> order;

            for (auto it = starts.rbegin(); it != starts.rend(); ++it) {
                if (!seen[*it]) {
                    std::vector<std::pair<int, std::size_t>> stack{{*it, 0}};
                    while (!stack.empty()) {
                        int u = stack.back().first;
                        auto &k = stack.back().second;
                        if (seen[u]) {
                            stack.pop_back();
                            continue;
                        }
                        if (k < partition.groups[u].preds.size()) {
                            int p = partition.groups[u].preds[k++];
                            if (!seen[p]) {
                                stack.emplace_back(p, 0);
                            }
                        } else {
                            seen[u] = 1;
                            order.push_back(u);
                            stack.pop_back();
                        }
                    }
                }
            }
            return order;
        }
        std::vector<int> deg(n), order;
        // 定义就绪堆的优先级，关键路径较长者先出堆。
        auto less = [&](int a, int b) {
            if (partition.groups[a].rank != partition.groups[b].rank) {
                return partition.groups[a].rank < partition.groups[b].rank;
            }
            return a > b;
        };
        std::priority_queue<int, std::vector<int>, decltype(less)> q(less);
        for (int i = 0; i < n; ++i) {
            deg[i] = static_cast<int>(partition.groups[i].preds.size());
            if (!deg[i]) {
                q.push(i);
            }
        }
        while (!q.empty()) {
            int u = q.top();
            q.pop();
            order.push_back(u);
            for (int v : partition.groups[u].succs) {
                if (!--deg[v]) {
                    q.push(v);
                }
            }
        }
        return order;
    }
    // 综合负载、依赖到达与通信代价贪心分核。
    std::vector<int> greedy(const std::vector<int> &order, double affinity) const {
        std::vector<int> assignment(partition.groups.size(), -1);
        std::vector<unsigned> touched(nets.size());
        std::array<std::array<double, NUM_PIPES>, 5> load{};
        std::vector<double> finish(partition.groups.size());
        double total = 0;
        for (const auto &a : partition.groups) {
            total += a.work;
        }
        double target = std::max(1.0, total / cores);
        for (int u : order) {
            const auto &a = partition.groups[u];
            int lower = 0;
            for (int p : a.preds) {
                lower = std::max(lower, assignment[p]);
            }
            int best = lower;
            double bestscore = std::numeric_limits<double>::infinity(), bestfinish = 0;
            for (int c = lower; c < cores; ++c) {
                double cut = 0, arrival = 0, peak = 0;
                for (int p : a.preds) {
                    arrival = std::max(arrival, finish[p] + (assignment[p] != c ? CROSS_CORE_DELAY : 0));
                }
                for (int ti : a.tensors) {
                    const auto &t = nets[ti];
                    if (t.producers.empty()) {
                        if (!(touched[ti] & (1u << c))) {
                            cut += t.bytes / DDR_BANDWIDTH;
                        }
                    } else {
                        for (int p : t.producers) {
                            if (assignment[p] >= 0 && assignment[p] != c) {
                                cut += 2 * t.bytes / DDR_BANDWIDTH;
                                break;
                            }
                        }
                    }
                }
                double end = arrival;
                for (int p = 0; p < NUM_PIPES; ++p) {
                    peak = std::max(peak, load[c][p] + a.load[p]);
                    if (a.load[p]) {
                        end = std::max(end, load[c][p] + a.load[p]);
                    }
                }
                double score = peak + 0.15 * end + affinity * cut;

                score += 0.15 * std::max(0.0, peak - target);
                if (score < bestscore - 1e-8) {
                    bestscore = score;
                    best = c;
                    bestfinish = end;
                }
            }
            assignment[u] = best;
            finish[u] = bestfinish;
            for (int p = 0; p < NUM_PIPES; ++p) {
                load[best][p] += a.load[p];
            }
            for (int ti : a.tensors) {
                touched[ti] |= 1u << best;
            }
        }
        return assignment;
    }
    // 按累计工作量中点将顺序切成连续核心区间。
    std::vector<int> cuts(const std::vector<int> &order) const {
        double total = 0;
        for (const auto &a : partition.groups) {
            total += std::max(1.0, a.work);
        }
        std::vector<int> a(partition.groups.size());
        double work = 0;
        for (int u : order) {
            double w = std::max(1.0, partition.groups[u].work);
            a[u] = std::min(cores - 1, static_cast<int>((work + w * 0.5) * cores / total));
            work += w;
        }
        return a;
    }
    // 比较快速估计或复核成绩，优先时间，其次搬运量。
    bool better(const Score &a, const Score &b) const {
        double eps = 1e-8 * std::max(1.0, b.time);
        return a.time < b.time - eps || (std::abs(a.time - b.time) <= eps && a.bytes < b.bytes - 0.5);
    }

    // 在就绪前沿内按缓存复用与释放收益重新排序。
    std::vector<int> locality_order(const Solution &plan, bool release_first) const {
        const int n = static_cast<int>(partition.groups.size());
        std::vector<int> position(n), degree(n), order;
        for (int i = 0; i < n; ++i) {
            position[plan.plan.order[i]] = i;
        }
        // 让参考顺序中更靠前的分组优先出堆。
        auto lower_priority = [&](int a, int b) { return position[a] > position[b]; };
        std::priority_queue<int, std::vector<int>, decltype(lower_priority)> frontier(lower_priority);
        std::vector<char> ready(n);
        for (int u = 0; u < n; ++u) {
            degree[u] = static_cast<int>(partition.groups[u].preds.size());
            if (!degree[u]) {
                ready[u] = true;
                frontier.push(u);
            }
        }
        std::vector<std::array<int, 5>> remaining(nets.size()), stamp(nets.size());
        std::vector<std::array<char, 5>> cached(nets.size());
        for (int t = 0; t < static_cast<int>(nets.size()); ++t) {
            for (int u : nets[t].touches) {
                ++remaining[t][plan.plan.core[u]];
            }
        }
        // 初始解复用排序中的小根堆，近似维护缓存访问先后。
        using CacheQueue = std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                                               std::greater<std::pair<int, int>>>;
        std::array<std::array<CacheQueue, 2>, 5> queues;
        std::array<std::array<double, 2>, 5> used{};
        std::array<int, 5> previous;
        previous.fill(-1);
        int clock = 0;
        while (!frontier.empty()) {
            std::vector<int> sampled, candidates;
            while (!frontier.empty() && sampled.size() < 48) {
                int u = frontier.top();
                frontier.pop();
                if (ready[u]) {
                    sampled.push_back(u);
                    candidates.push_back(u);
                }
            }
            if (sampled.empty()) {
                break;
            }
            for (int c = 0; c < cores; ++c) {
                if (previous[c] >= 0) {
                    int inspected = 0;
                    for (int t : partition.groups[previous[c]].tensors) {
                        if (!cached[t][c]) {
                            continue;
                        }
                        for (int u : nets[t].touches) {
                            if (++inspected > 512) {
                                break;
                            }
                            if (ready[u] && plan.plan.core[u] == c) {
                                candidates.push_back(u);
                            }
                        }
                        if (inspected > 512) {
                            break;
                        }
                    }
                }
            }
            unique(candidates);
            int selected = sampled.front();
            double best_score = -std::numeric_limits<double>::infinity();
            for (int u : candidates) {
                int c = plan.plan.core[u];
                double reuse = 0, freed = 0, incoming = 0;
                for (int t : partition.groups[u].tensors) {
                    const auto &net = nets[t];
                    if (net.touches.size() < 2) {
                        continue;
                    }
                    double weight = net.bytes / DDR_BANDWIDTH;
                    if (release_first) {
                        weight *= net.memory == 1 ? 4.0 : 1.0;
                    }
                    if (cached[t][c]) {
                        reuse += weight;
                        if (remaining[t][c] == 1) {
                            freed += weight;
                        }
                    } else if (remaining[t][c] > 1) {
                        incoming += weight;
                    }
                }
                double score =
                    reuse + (release_first ? 1.5 : 0.25) * freed - (release_first ? 0.3 : 0.05) * incoming;

                score += 0.02 * std::min(partition.groups[u].rank, 20000.0);
                if (score > best_score + 1e-8 ||
                    (std::abs(score - best_score) <= 1e-8 && position[u] < position[selected])) {
                    best_score = score;
                    selected = u;
                }
            }
            for (int u : sampled) {
                if (u != selected) {
                    frontier.push(u);
                }
            }
            ready[selected] = false;
            order.push_back(selected);
            int c = plan.plan.core[selected];
            previous[c] = selected;
            for (int t : partition.groups[selected].tensors) {
                const auto &net = nets[t];
                --remaining[t][c];
                if (remaining[t][c] == 0) {
                    if (cached[t][c]) {
                        used[c][net.memory] -= net.bytes;
                        cached[t][c] = false;
                    }
                    continue;
                }
                if (!cached[t][c]) {
                    cached[t][c] = true;
                    used[c][net.memory] += net.bytes;
                }
                stamp[t][c] = ++clock;
                queues[c][net.memory].emplace(clock, t);
            }
            for (int m = 0; m < 2; ++m) {

                double capacity = 0.75 * (m == 0 ? L1_CAPACITY : UB_CAPACITY);
                auto &queue = queues[c][m];
                while (used[c][m] > capacity && !queue.empty()) {
                    auto entry = queue.top();
                    queue.pop();
                    int t = entry.second;
                    if (!cached[t][c] || stamp[t][c] != entry.first) {
                        continue;
                    }
                    cached[t][c] = false;
                    used[c][m] -= nets[t].bytes;
                }
            }
            for (int v : partition.groups[selected].succs) {
                if (!--degree[v]) {
                    ready[v] = true;
                    frontier.push(v);
                }
            }
        }
        if (static_cast<int>(order.size()) != n) {
            throw NoPlan("ordering frontier did not cover the DAG");
        }
        return order;
    }

    // 将共享数据的分组向同核前一使用者附近插入。
    Solution insertion_candidate(const Solution &base) {
        Solution best = base;
        Score score = estimate_cost(base.plan.core, base.plan.order);
        best.score.makespan = score.time;
        best.score.copy_bytes = score.bytes;
        const int n = static_cast<int>(base.plan.order.size());
        std::vector<int> position(n);
        for (int i = 0; i < n; ++i) {
            position[base.plan.order[i]] = i;
        }
        std::vector<int> tensors(nets.size());
        std::iota(tensors.begin(), tensors.end(), 0);
        // 按字节数与使用组数乘积优先检查共享张量。
        std::stable_sort(tensors.begin(), tensors.end(), [&](int a, int b) {
            return nets[a].bytes * nets[a].touches.size() > nets[b].bytes * nets[b].touches.size();
        });
        int trials = 0;
        for (int t : tensors) {
            if (trials >= 16) {
                break;
            }
            if (nets[t].touches.size() < 2) {
                continue;
            }
            auto users = nets[t].touches;
            // 按原方案位置排列共享张量使用者。
            std::sort(users.begin(), users.end(), [&](int a, int b) { return position[a] < position[b]; });
            std::array<int, 5> previous;
            previous.fill(-1);
            for (int u : users) {
                int c = base.plan.core[u], anchor = previous[c];
                previous[c] = u;
                if (anchor < 0 || trials >= 16) {
                    continue;
                }
                int target = position[anchor] + 1, old = position[u];
                if (target >= old) {
                    continue;
                }
                bool legal = true;
                for (int p : partition.groups[u].preds) {
                    if (position[p] >= target) {
                        legal = false;
                        break;
                    }
                }
                if (!legal) {
                    continue;
                }
                ++trials;
                Solution candidate = base;
                candidate.plan.order.erase(candidate.plan.order.begin() + old);
                candidate.plan.order.insert(candidate.plan.order.begin() + target, u);
                auto next = estimate_cost(candidate.plan.core, candidate.plan.order);
                if (better(next, score)) {
                    score = next;
                    best = std::move(candidate);
                    best.score.makespan = score.time;
                    best.score.copy_bytes = score.bytes;
                }
            }
        }
        return best;
    }

  public:
    // 准备初始化要反复查询的数据，不在构造器里选方案。
    // op_tensors 记录每个操作访问哪些张量；nets 将张量读写者转换为分组编号。
    // 后续据此估算共享输入、跨核搬运和数据驻留，避免每次重新遍历原图关系。
    InitialPlanBuilder(const Problem &graph, const Partition &groups, int n)
        : g(graph), partition(groups), cores(n), op_tensors(g.ops.size()) {
        for (int u : g.op_topo) {
            op_tensors[u] = g.ops[u].inputs;
            op_tensors[u].insert(op_tensors[u].end(), g.ops[u].outputs.begin(), g.ops[u].outputs.end());
            unique(op_tensors[u]);
        }
        for (const auto &t : g.tensors) {
            Net net;
            net.bytes = static_cast<double>(t.size);
            net.memory = t.memory == 1 ? 0 : 1;
            net.output = t.output;
            for (int u : t.producers) {
                net.producers.push_back(partition.op_group[u]);
            }
            for (int u : t.consumers) {
                net.consumers.push_back(partition.op_group[u]);
            }
            unique(net.producers);
            unique(net.consumers);
            net.touches = net.producers;
            net.touches.insert(net.touches.end(), net.consumers.begin(), net.consumers.end());
            unique(net.touches);
            nets.push_back(std::move(net));
        }
    }
    // 在固定初始分组上，比较多种分核和顺序，选择后续搜索的起点。
    // 先枚举实际使用的核心数，再组合三种遍历顺序与两类分核方式。
    // 粗估用于记录备选方案；去重后的候选全部完整复核，粗估最优不等于最终最优。
    // 对已有方案再试数据复用排序和局部插入调整，最后返回所有核心数中的最优方案。
    // 此处返回分核、顺序和成绩；外层 generate_initial_plan() 附上对应的完整分组。
    Solution run() {
        const int requested_cores = cores;
        Solution best;
        best.score.makespan = std::numeric_limits<double>::infinity();
        bool found = false;

        // 1. 多用核心可能增加搬运，因此也比较少用核心的方案，空闲核心在输出中保留。
        for (int active = 1; active <= requested_cores; ++active) {
            cores = active;
            Solution fast_best;
            fast_best.score.makespan = std::numeric_limits<double>::infinity();
            std::vector<Solution> candidates;
            // 快速评估候选、更新快速最优并去重入池。
            auto add = [&](std::vector<int> assignment, const std::vector<int> &order) {
                auto score = estimate_cost(assignment, order);
                Solution candidate;
                candidate.plan = {std::move(assignment), order};
                candidate.score = {score.time, score.bytes};
                if (fast_best.plan.core.empty() ||
                    better(score, {fast_best.score.makespan, fast_best.score.copy_bytes})) {
                    fast_best = candidate;
                }
                for (const auto &existing : candidates) {
                    if (existing.plan.core == candidate.plan.core &&
                        existing.plan.order == candidate.plan.order) {
                        return;
                    }
                }
                candidates.push_back(std::move(candidate));
            };
            // 2. 同一遍历顺序下，比较偏重共享数据的逐组分核与按累计计算量切段。
            // affinity 控制逐组分核对数据共享的重视程度；具体评分在 greedy() 中。
            for (int mode = 0; mode < 3; ++mode) {
                auto order = order_for(mode);
                for (double affinity : {0.25, 1.0, 4.0}) {
                    add(greedy(order, affinity), order);
                }
                add(cuts(order), order);
            }
            add(std::vector<int>(partition.groups.size(), 0), order_for(1));
            Solution reviewed;
            reviewed.score.makespan = std::numeric_limits<double>::infinity();
            bool valid = false;
            // 用完整复核成绩更新当前核心数与全局最优方案。
            auto select = [&](Solution candidate, const PlanEvaluation &result) {
                if (!result.valid) {
                    return;
                }
                Score score{result.makespan, result.copy_bytes};
                candidate.score.makespan = score.time;
                candidate.score.copy_bytes = score.bytes;
                if (!valid || better(score, {reviewed.score.makespan, reviewed.score.copy_bytes})) {
                    reviewed = candidate;
                    valid = true;
                }
                if (!found || better(score, {best.score.makespan, best.score.copy_bytes})) {
                    best = std::move(candidate);
                    found = true;
                }
            };
            // 完整复核单个排序候选并送入统一选优。
            auto review = [&](Solution candidate) {
                auto result = evaluate_plan(g, partition.op_group, candidate.plan, active);
                select(std::move(candidate), result);
            };
            // 3. 完整模拟核内调度、容量换出、跨核等待和共享带宽，淘汰不可行方案。
            // 并行只计算成绩，仍按候选原顺序选优，不受线程完成先后影响。
            std::vector<PlanEvaluation> results(candidates.size());
            // 独立复核指定序号候选，供线程池调用。
            parallel_for(candidates.size(), g.op_topo.size() >= 256, [&](std::size_t i) {
                results[i] = evaluate_plan(g, partition.op_group, candidates[i].plan, active);
            });
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                select(candidates[i], results[i]);
            }
            if (!valid) {
                continue;
            }

            // 4. 分核保持不变，再调整组的执行顺序；同时保留粗估起点，避免只看一种顺序。
            // 两种 locality_order 分别侧重复用与释放数据；插入调整仍须完整复核才接受。
            std::vector<Solution> order_seeds{reviewed};
            if (reviewed.plan.core != fast_best.plan.core || reviewed.plan.order != fast_best.plan.order) {
                order_seeds.push_back(fast_best);
            }
            for (const auto &seed : order_seeds) {
                for (bool release_first : {false, true}) {
                    Solution candidate = seed;
                    candidate.plan.order = locality_order(seed, release_first);
                    review(std::move(candidate));
                }
            }
            review(insertion_candidate(reviewed));
        }
        cores = requested_cores;
        if (!found) {
            throw NoPlan("no candidate passed internal execution review");
        }
        best.statistics.evaluations = evaluations;

        for (int u : partition.topo) {
            for (int v : partition.groups[u].succs) {
                if (best.plan.core[u] > best.plan.core[v]) {
                    throw NoPlan("internal core dependency invariant failed");
                }
            }
        }
        return best;
    }
};
}
// 使用 build_groups() 给出的固定分组，生成一个可供后续改进的起点。
// Builder 先汇总张量关系，再枚举分核、比较完整模拟结果并尝试局部重排。
// 返回的 Solution 同时持有分组、核心归属、执行顺序和内部成绩；找不到可行方案则抛异常。
// 这里不会把初始分组当作最终约束，后续 improve_plan() 可以重新划分。
Solution generate_initial_plan(const Problem &graph, const Partition &partition, int cores) {
    auto solution = InitialPlanBuilder(graph, partition, cores).run();
    solution.partition = partition;
    return solution;
}
}
