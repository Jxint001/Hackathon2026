// 解析并校验计算图，收缩 COPY 依赖，绑定共享写者并建立无环分组。
#include "graph.hpp"
#include "json_parser.hpp"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>

namespace npu {
namespace {
// 排序并去除容器中的重复值。
template <class T> void unique(std::vector<T> &x) {
    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
}
// 初始分组阶段的并查集，用于共享写者绑定、强连通分量合并和短链收缩。
struct DSU {
    std::vector<int> p, size;
    // 初始化并查集，每个节点独立成组。
    explicit DSU(int n) : p(n), size(n, 1) {
        std::iota(p.begin(), p.end(), 0);
    }
    // 查找节点所在集合并压缩路径。
    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    // 按组大小合并集合，平局时使用索引确定顺序。
    void join(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (size[a] < size[b] || (size[a] == size[b] && a > b)) {
            std::swap(a, b);
        }
        p[b] = a;
        size[a] += size[b];
    }
};
// 读取 JSON 非负整数。
Id nonnegative(const Json &v) {
    Id n = v.integer();
    if (n < 0) {
        throw InputError("expected nonnegative integer");
    }
    return n;
}
// 使用小索引优先的拓扑排序检测有向环。
std::vector<int> topological(const std::vector<std::vector<int>> &adj) {
    std::vector<int> deg(adj.size()), order;
    for (const auto &vs : adj) {
        for (int v : vs) {
            ++deg[v];
        }
    }
    std::priority_queue<int, std::vector<int>, std::greater<int>> q;
    for (int i = 0; i < static_cast<int>(adj.size()); ++i) {
        if (!deg[i]) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u = q.top();
        q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            if (!--deg[v]) {
                q.push(v);
            }
        }
    }
    if (order.size() != adj.size()) {
        throw InputError("input graph contains a cycle");
    }
    return order;
}
}
// 解析官方图格式，建立计算操作依赖并检查单操作容量。
Problem read_graph(const Json &root) {
    Problem g;
    std::unordered_map<Id, int> oi, ti;
    for (const auto &x : root.at("ops").items()) {
        Op o;
        o.id = nonnegative(x.at("id"));
        auto name = x.at("op").string();
        o.copy = name == "COPY_IN" ? 1 : (name == "COPY_OUT" ? 2 : 0);
        const auto &pipe = x.at("pipe").string();
        const std::array<std::string, NUM_PIPES> names = {"PIPE_M", "PIPE_V", "PIPE_MTE2", "PIPE_MTE3",
                                                          "PIPE_FIX"};
        auto it = std::find(names.begin(), names.end(), pipe);
        if (it == names.end()) {
            throw InputError("unsupported pipe: " + pipe);
        }
        o.pipe = static_cast<int>(it - names.begin());
        auto cycles = x.at("cycles").integer();
        if (!o.copy && cycles < 0) {
            throw InputError("negative compute cycles");
        }
        o.cycles = static_cast<double>(std::max<Id>(1, cycles));
        g.ops.push_back(std::move(o));
    }
    // 比较操作 id，固定内部操作索引顺序。
    std::sort(g.ops.begin(), g.ops.end(), [](const Op &a, const Op &b) { return a.id < b.id; });
    for (int i = 0; i < static_cast<int>(g.ops.size()); ++i) {
        if (!oi.emplace(g.ops[i].id, i).second) {
            throw InputError("duplicate op id");
        }
    }
    for (const auto &x : root.at("tensors").items()) {
        Tensor t;
        t.id = nonnegative(x.at("id"));
        t.size = nonnegative(x.at("size"));
        auto pos = x.at("pos").string();
        if (pos == "DDR") {
            t.memory = 0;
        } else if (pos == "L1") {
            t.memory = 1;
        } else if (pos == "UB") {
            t.memory = 2;
        } else {
            throw InputError("unsupported tensor memory");
        }
        if (oi.count(t.id)) {
            throw InputError("duplicate node id");
        }
        g.tensors.push_back(std::move(t));
    }
    // 比较张量 id，固定内部张量索引顺序。
    std::sort(g.tensors.begin(), g.tensors.end(),
              [](const Tensor &a, const Tensor &b) { return a.id < b.id; });
    for (int i = 0; i < static_cast<int>(g.tensors.size()); ++i) {
        if (!ti.emplace(g.tensors[i].id, i).second) {
            throw InputError("duplicate tensor id");
        }
    }
    std::vector<std::vector<int>> adj(g.ops.size());
    for (const auto &x : root.at("edges").items()) {
        Id a = nonnegative(x.at("source")), b = nonnegative(x.at("target"));
        auto ao = oi.find(a), bo = oi.find(b);
        auto at = ti.find(a), bt = ti.find(b);
        if (ao != oi.end() && bt != ti.end()) {
            g.ops[ao->second].outputs.push_back(bt->second);
            g.tensors[bt->second].producers.push_back(ao->second);
        } else if (at != ti.end() && bo != oi.end()) {
            g.ops[bo->second].inputs.push_back(at->second);
            g.tensors[at->second].consumers.push_back(bo->second);
        } else {
            throw InputError("expected tensor-op edge with existing endpoints");
        }
    }
    for (auto &t : g.tensors) {
        unique(t.producers);
        unique(t.consumers);
        for (int u : t.producers) {
            for (int v : t.consumers) {
                adj[u].push_back(v);
            }
        }
        for (int u : t.consumers) {
            if (g.ops[u].copy == 2) {
                t.output = true;
            }
        }
    }
    for (auto &vs : adj) {
        unique(vs);
    }
    auto order = topological(adj);
    for (int u : order) {
        if (!g.ops[u].copy) {
            g.op_topo.push_back(u);
        }
    }
    std::vector<std::vector<int>> pred(g.ops.size()), front(g.ops.size());
    for (int u : order) {
        for (int v : adj[u]) {
            pred[v].push_back(u);
        }
    }
    for (int u : order) {
        std::vector<int> ps;
        for (int p : pred[u]) {
            ps.insert(ps.end(), front[p].begin(), front[p].end());
        }
        unique(ps);
        if (g.ops[u].copy) {
            front[u] = std::move(ps);
        } else {
            g.ops[u].preds = ps;
            for (int p : ps) {
                g.ops[p].succs.push_back(u);
            }
            front[u] = {u};
        }
    }
    for (auto &o : g.ops) {
        unique(o.inputs);
        unique(o.outputs);
    }
    for (auto &t : g.tensors) {
        // 识别需要从计算读写者列表中剔除的 COPY 操作。
        auto copy = [&](int o) { return g.ops[o].copy != 0; };
        t.producers.erase(std::remove_if(t.producers.begin(), t.producers.end(), copy), t.producers.end());
        t.consumers.erase(std::remove_if(t.consumers.begin(), t.consumers.end(), copy), t.consumers.end());
        if (!t.producers.empty() && t.consumers.empty()) {
            t.output = true;
        }
    }

    for (const auto &o : g.ops) {
        if (!o.copy) {
            auto ts = o.inputs;
            ts.insert(ts.end(), o.outputs.begin(), o.outputs.end());
            unique(ts);
            std::array<long double, 3> bytes{};
            for (int t : ts) {
                bytes[g.tensors[t].memory == 1 ? 1 : 2] += g.tensors[t].size;
            }
            if (bytes[1] > L1_CAPACITY || bytes[2] > UB_CAPACITY) {
                throw NoPlan("operation working set exceeds fixed local capacity");
            }
        }
    }
    return g;
}

// 输入是各组包含的操作列表；此处不决定分核，也不改变组内给定顺序。
// 先建立操作到组的映射，再汇总依赖、涉及张量和各管道计算量，最后求拓扑序与后续工作量。
// 初始化和搜索都调用这里；返回的 Partition 已完整，不需要 finish() 再补字段。
// 重复、遗漏的操作或成环分组不能作为完整方案返回。
Partition make_partition(const Problem &problem, std::vector<std::vector<int>> group_ops) {
    Partition partition;
    partition.groups.resize(group_ops.size());
    partition.op_group.assign(problem.ops.size(), -1);
    for (int index = 0; index < static_cast<int>(group_ops.size()); ++index) {
        if (group_ops[index].empty()) {
            throw NoPlan("empty group");
        }
        auto &group = partition.groups[index];
        group.ops = std::move(group_ops[index]);
        for (int op : group.ops) {
            if (op < 0 || op >= static_cast<int>(problem.ops.size()) || problem.ops[op].copy ||
                partition.op_group[op] != -1) {
                throw NoPlan("invalid group membership");
            }
            partition.op_group[op] = index;
        }
    }
    // 按输入操作索引累计，保留初始化的浮点累加顺序。
    for (int op = 0; op < static_cast<int>(problem.ops.size()); ++op) {
        if (problem.ops[op].copy) {
            continue;
        }
        int index = partition.op_group[op];
        if (index < 0) {
            throw NoPlan("ungrouped compute operation");
        }
        auto &group = partition.groups[index];
        const auto &operation = problem.ops[op];
        group.load[operation.pipe] += operation.cycles;
        group.tensors.insert(group.tensors.end(), operation.inputs.begin(), operation.inputs.end());
        group.tensors.insert(group.tensors.end(), operation.outputs.begin(), operation.outputs.end());
        for (int next : operation.succs) {
            int target = partition.op_group[next];
            if (target < 0) {
                throw NoPlan("ungrouped successor");
            }
            if (index != target) {
                group.succs.push_back(target);
                partition.groups[target].preds.push_back(index);
            }
        }
    }
    std::vector<std::vector<int>> adjacency(partition.groups.size());
    for (std::size_t index = 0; index < partition.groups.size(); ++index) {
        auto &group = partition.groups[index];
        unique(group.preds);
        unique(group.succs);
        unique(group.tensors);
        group.work = *std::max_element(group.load.begin(), group.load.end());
        adjacency[index] = group.succs;
    }
    partition.topo = topological(adjacency);
    for (auto it = partition.topo.rbegin(); it != partition.topo.rend(); ++it) {
        auto &group = partition.groups[*it];
        for (int next : group.succs) {
            group.rank = std::max(group.rank, partition.groups[next].rank);
        }
        group.rank += group.work;
    }
    return partition;
}

// 为初始化建立较少、便于分核的计算组；返回完整 Partition，不修改输入图。
// 先合并共享写者以满足同核约束，再合并因收缩产生的组间环，最后合并短的连续依赖链。
// 前两步处理约束，短链合并是为了减少候选构造的规模，并非题目强制要求。
// 最后按原图拓扑顺序整理各组操作，交给 make_partition() 统一生成分组信息。
// 这些只是初始计算组，后续搜索允许重新切分，不要求一个组始终连续执行。
Partition build_groups(const Problem &g) {
    const int n = static_cast<int>(g.ops.size());
    DSU d(n);

    // 1. 合并同一张量的写者，保证它们同核。
    for (const auto &t : g.tensors) {
        for (std::size_t i = 1; i < t.producers.size(); ++i) {
            d.join(t.producers[0], t.producers[i]);
        }
    }

    // 2. 建立组间依赖，合并成环的组。
    std::vector<std::vector<int>> adj(n), rev(n);
    for (int u = 0; u < n; ++u) {
        if (!g.ops[u].copy) {
            for (int v : g.ops[u].succs) {
                int a = d.find(u), b = d.find(v);
                if (a != b) {
                    adj[a].push_back(b);
                    rev[b].push_back(a);
                }
            }
        }
    }
    for (auto &v : adj) {
        unique(v);
    }
    for (auto &v : rev) {
        unique(v);
    }
    std::vector<char> seen(n);
    std::vector<int> finish;
    for (int s = 0; s < n; ++s) {
        if (!g.ops[s].copy && d.find(s) == s && !seen[s]) {
            std::vector<std::pair<int, std::size_t>> st{{s, 0}};
            seen[s] = 1;
            while (!st.empty()) {
                int u = st.back().first;
                auto &j = st.back().second;
                if (j < adj[u].size()) {
                    int v = adj[u][j++];
                    if (!seen[v]) {
                        seen[v] = 1;
                        st.emplace_back(v, 0);
                    }
                } else {
                    finish.push_back(u);
                    st.pop_back();
                }
            }
        }
    }
    std::fill(seen.begin(), seen.end(), 0);
    for (auto it = finish.rbegin(); it != finish.rend(); ++it) {
        if (!seen[*it]) {
            int root = *it;
            std::vector<int> st{root};
            seen[root] = 1;
            while (!st.empty()) {
                int u = st.back();
                st.pop_back();
                d.join(root, u);
                for (int v : rev[u]) {
                    if (!seen[v]) {
                        seen[v] = 1;
                        st.push_back(v);
                    }
                }
            }
        }
    }
    // 3. 重建组间依赖，合并单出边、单入边的短链（仅合并总数不超过 64 的组）。
    adj.assign(n, {});
    rev.assign(n, {});
    for (int u = 0; u < n; ++u) {
        if (!g.ops[u].copy) {
            for (int v : g.ops[u].succs) {
                int a = d.find(u), b = d.find(v);
                if (a != b) {
                    adj[a].push_back(b);
                    rev[b].push_back(a);
                }
            }
        }
    }
    for (auto &v : adj) {
        unique(v);
    }
    for (auto &v : rev) {
        unique(v);
    }

    for (int u = 0; u < n; ++u) {
        if (adj[u].size() == 1) {
            int v = adj[u][0];
            int a = d.find(u), b = d.find(v);
            if (rev[v].size() == 1 && a != b && d.size[a] + d.size[b] <= 64) {
                d.join(a, b);
            }
        }
    }
    // 收集操作列表，再统一构造完整分组。
    std::vector<std::vector<int>> group_ops;
    std::unordered_map<int, int> index;
    for (int u = 0; u < n; ++u) {
        if (!g.ops[u].copy) {
            int root = d.find(u);
            auto inserted = index.emplace(root, static_cast<int>(group_ops.size()));
            if (inserted.second) {
                group_ops.emplace_back();
            }
            group_ops[inserted.first->second].push_back(u);
        }
    }
    std::vector<int> op_position(n);
    for (int i = 0; i < static_cast<int>(g.op_topo.size()); ++i) {
        op_position[g.op_topo[i]] = i;
    }
    // 初始分组保持原图拓扑顺序；搜索候选保持其提交顺序。
    for (auto &ops : group_ops) {
        std::sort(ops.begin(), ops.end(), [&](int a, int b) { return op_position[a] < op_position[b]; });
    }
    return make_partition(g, std::move(group_ops));
}
}
