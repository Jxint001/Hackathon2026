#pragma once

#include "graph.hpp"
#include <numeric>

namespace npu {

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int a) { while (a != parent[a]) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
    void join(int a, int b) { a = find(a); b = find(b); if (a != b) parent[std::max(a, b)] = std::min(a, b); }
};

// 原图事实：同核约束不限制操作必须连续执行。
struct GraphAnalysis {
    std::vector<int> co_group, depth, reader_count, writer_count;
    std::vector<std::vector<int>> co_groups;
    explicit GraphAnalysis(const Graph& graph);
};

double median_position(std::vector<double> values);
std::vector<int> reference_order_with_copies(const Graph& graph);
std::vector<double> infer_locality_axis(const Graph& graph);

// 用于构造候选的操作组，最终仍逐个操作输出。
struct OpGroup {
    std::vector<int> ops, pred, succ, inputs, outputs, uses, nets;
    std::array<Cost, kPipeCount> work{};
    int depth = 0, pipe = 0;
    double coordinate = -1;
};

struct TensorLinks {
    std::vector<int> producers, consumers;
    Cost service = 0;
    bool output = false;
};

struct SchedulingGroups {
    const Graph& graph;
    const GraphAnalysis& facts;
    std::vector<OpGroup> units;
    std::vector<int> of, topo, dfs;
    std::vector<TensorLinks> nets;
    std::vector<std::vector<int>> chains;
    bool spatial = false, compact;

    SchedulingGroups(const Graph& graph, const GraphAnalysis& analysis, bool use_compact = false);
    void prepare_order(bool use_compact);
    void add_net(TensorLinks net);
    std::vector<int> normalize_owners(std::vector<int> owner) const;
    std::vector<std::pair<int, int>> close_changes(const std::vector<std::pair<int, int>>& changes) const;
    Plan project(const std::vector<int>& owner, const std::vector<int>& order, int cores) const;
    std::vector<int> owners(const Plan& plan) const;
};

} // npu 命名空间
