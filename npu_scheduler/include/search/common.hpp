#pragma once
#include "plan_evaluator.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <tuple>

namespace npu {
namespace search_detail {
using Sequences = std::vector<std::vector<int>>;
template <class T> void distinct(std::vector<T> &v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}
// 搜索阶段的共享写者并查集，确定必须同时迁核的操作集合。
struct UnionFind {
    std::vector<int> parent;
    // 初始化用于共享写者闭包的并查集。
    explicit UnionFind(int n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    // 查找共享写者集合代表并压缩路径。
    int find(int u) {
        while (u != parent[u]) {
            parent[u] = parent[parent[u]];
            u = parent[u];
        }
        return u;
    }
    // 合并两个共享写者集合，选择较小根索引。
    void join(int u, int v) {
        u = find(u);
        v = find(v);
        if (u > v) {
            std::swap(u, v);
        }
        parent[v] = u;
    }
};
// 从两类共享输入和等长独立计算链识别出的二维计算网格。
struct Grid {
    int rows = 0, cols = 0, depth = 0;
    std::vector<std::vector<int>> cells;
};
// 计算数值序列的中位数。
double median(std::vector<double> v);
std::vector<double> locality_axis(const Problem &g);
}
}
