#include "evaluation/model.hpp"
namespace npu {
namespace evaluation_detail {
void local_order(Task &task, const std::vector<int> &group_rank) {
    build_views(task);
    int n = static_cast<int>(task.ops.size());
    std::vector<int> degree(n), depth(n), queue;
    for (int u = 0; u < n; ++u) {
        degree[u] = static_cast<int>(task.ops[u].preds.size());
        if (!degree[u]) {
            queue.push_back(u);
        }
    }
    for (std::size_t j = 0; j < queue.size(); ++j) {
        int u = queue[j];
        for (int v : task.ops[u].succs) {
            depth[v] = std::max(depth[v], depth[u] + 1);
            if (!--degree[v]) {
                queue.push_back(v);
            }
        }
    }
    if (static_cast<int>(queue.size()) != n) {
        throw InvalidReview{};
    }
    // 实现逆向 DFS 的 COPY 类型、深度及 id 平局优先级。
    auto less = [&](int a, int b, bool start) {
        const auto &x = task.ops[a];
        const auto &y = task.ops[b];
        bool ordinary_a = x.copy != (start ? 2 : 1);
        bool ordinary_b = y.copy != (start ? 2 : 1);
        if (ordinary_a != ordinary_b) {
            return ordinary_a < ordinary_b;
        }
        if (depth[a] != depth[b]) {
            return depth[a] < depth[b];
        }
        return x.id > y.id;
    };
    std::vector<int> stack, parents;
    for (int u = 0; u < n; ++u) {
        if (task.ops[u].succs.empty()) {
            stack.push_back(u);
        }
    }
    // 调用起点模式的比较器排列 DFS 末端操作。
    std::sort(stack.begin(), stack.end(), [&](int a, int b) { return less(a, b, true); });
    std::vector<char> seen(n);
    while (!stack.empty()) {
        int u = stack.back();
        if (seen[u]) {
            stack.pop_back();
            continue;
        }
        parents.clear();
        for (int p : task.ops[u].preds) {
            if (!seen[p]) {
                parents.push_back(p);
            }
        }
        if (parents.empty()) {
            seen[u] = true;
            task.sequence.push_back(u);
            stack.pop_back();
        } else {
            // 调用前驱模式的比较器排列 DFS 未访问前驱。
            std::sort(parents.begin(), parents.end(), [&](int a, int b) { return less(a, b, false); });
            stack.insert(stack.end(), parents.begin(), parents.end());
        }
    }
    // 按提交的子图顺序对指令稳定排序。
    std::stable_sort(task.sequence.begin(), task.sequence.end(), [&](int a, int b) {
        return group_rank[task.ops[a].group] < group_rank[task.ops[b].group];
    });
    std::vector<int> position(n);
    for (int i = 0; i < n; ++i) {
        position[task.sequence[i]] = i;
    }
    for (int u = 0; u < n; ++u) {
        for (int p : task.ops[u].preds) {
            if (position[p] >= position[u]) {
                throw InvalidReview{};
            }
        }
    }
}

}
}
