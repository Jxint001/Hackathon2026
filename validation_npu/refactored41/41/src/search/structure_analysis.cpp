#include "search/common.hpp"
namespace npu {
namespace search_detail {
double median(std::vector<double> v) {
    if (v.empty()) {
        return -1;
    }
    std::sort(v.begin(), v.end());
    return (v[(v.size() - 1) / 2] + v[v.size() / 2]) * 0.5;
}

// 从共享输入的重复重叠关系推导空间位置轴。
std::vector<double> locality_axis(const Problem &g) {
    std::vector<int> level(g.ops.size());
    std::vector<int> first;
    // 取各依赖路径上首批 cycles > 1 的 PIPE_M 操作。
    for (int v : g.op_topo) {
        for (int p : g.ops[v].preds) {
            level[v] = std::max(level[v], level[p]);
        }
        if (g.ops[v].pipe == 0 && g.ops[v].cycles > 1) {
            if (!level[v]) {
                first.push_back(v);
            }
            ++level[v];
        }
    }
    std::vector<int> frequency(g.tensors.size());
    for (int v : first) {
        for (int t : g.ops[v].inputs) {
            ++frequency[t];
        }
    }
    // 按周期数和高频共享输入分组，避免混合不同计算族。
    std::map<std::pair<double, std::vector<int>>, std::vector<int>> families;
    for (int v : first) {
        std::vector<int> common;
        for (int t : g.ops[v].inputs) {
            if (frequency[t] >= std::max(4, static_cast<int>(first.size()) / 20)) {
                common.push_back(t);
            }
        }
        families[{g.ops[v].cycles, common}].push_back(v);
    }
    std::vector<int> best_path;
    for (const auto &family : families) {
        const auto &ops = family.second;
        if (ops.size() < 4) {
            continue;
        }
        std::map<int, int> count;
        for (int v : ops) {
            for (int t : g.ops[v].inputs) {
                ++count[t];
            }
        }
        // 保留局部复用输入，排除仅用一次或大范围共享的输入。
        std::set<int> local;
        for (const auto &item : count) {
            if (item.second > 1 && item.second * 2 < static_cast<int>(ops.size())) {
                local.insert(item.first);
            }
        }
        if (local.size() <= best_path.size()) {
            continue;
        }
        std::map<std::pair<int, int>, int> pairs;
        int strongest = 0;
        for (int v : ops) {
            std::vector<int> inputs;
            for (int t : g.ops[v].inputs) {
                if (local.count(t)) {
                    inputs.push_back(t);
                }
            }
            // 过宽的输入集合不用于推断滑动窗口。
            if (inputs.size() > 32) {
                continue;
            }
            for (std::size_t a = 0; a < inputs.size(); ++a) {
                for (std::size_t b = a + 1; b < inputs.size(); ++b) {
                    strongest = std::max(strongest, ++pairs[{inputs[a], inputs[b]}]);
                }
            }
        }
        // 最常共同使用的输入视为邻居，要求连成一条完整路径。
        std::map<int, std::vector<int>> adjacent;
        for (const auto &edge : pairs) {
            if (edge.second == strongest) {
                adjacent[edge.first.first].push_back(edge.first.second);
                adjacent[edge.first.second].push_back(edge.first.first);
            }
        }
        std::vector<int> endpoints;
        bool path_graph = adjacent.size() == local.size();
        for (const auto &item : adjacent) {
            if (item.second.size() == 1) {
                endpoints.push_back(item.first);
            } else if (item.second.size() != 2) {
                path_graph = false;
            }
        }
        if (!path_graph || endpoints.size() != 2) {
            continue;
        }
        std::vector<int> path;
        int previous = -1, v = endpoints[0];
        while (v >= 0 && path.size() <= local.size()) {
            path.push_back(v);
            int next = -1;
            for (int u : adjacent[v]) {
                if (u != previous) {
                    next = u;
                }
            }
            previous = v;
            v = next;
        }
        if (v < 0 && path.size() == local.size()) {
            best_path = std::move(path);
        }
    }
    // 未识别到路径时保持 -1，后续使用无位置轴的分核候选。
    std::vector<double> coord(g.ops.size(), -1), tensor_coord(g.tensors.size(), -1);
    for (int i = 0; i < static_cast<int>(best_path.size()); ++i) {
        tensor_coord[best_path[i]] = i;
    }
    // 先向输入预处理节点反传位置。
    for (auto it = g.op_topo.rbegin(); it != g.op_topo.rend(); ++it) {
        const int v = *it;
        if (level[v]) {
            continue;
        }
        std::vector<double> values;
        for (int t : g.ops[v].outputs) {
            if (tensor_coord[t] >= 0) {
                values.push_back(tensor_coord[t]);
            }
        }
        coord[v] = median(std::move(values));
        if (coord[v] >= 0) {
            for (int t : g.ops[v].inputs) {
                if (tensor_coord[t] < 0) {
                    tensor_coord[t] = coord[v];
                }
            }
        }
    }
    // 只保留路径锚点，再按输入位置中位数向后传播。
    std::fill(tensor_coord.begin(), tensor_coord.end(), -1);
    for (int i = 0; i < static_cast<int>(best_path.size()); ++i) {
        tensor_coord[best_path[i]] = i;
    }
    for (int v : g.op_topo) {
        if (coord[v] < 0) {
            std::vector<double> values;
            for (int t : g.ops[v].inputs) {
                if (tensor_coord[t] >= 0) {
                    values.push_back(tensor_coord[t]);
                    continue;
                }
                std::vector<double> writers;
                for (int p : g.tensors[t].producers) {
                    if (!g.ops[p].copy && coord[p] >= 0) {
                        writers.push_back(coord[p]);
                    }
                }
                if (!writers.empty()) {
                    values.push_back(median(std::move(writers)));
                }
            }
            if (values.empty()) {
                for (int p : g.ops[v].preds) {
                    if (coord[p] >= 0) {
                        values.push_back(coord[p]);
                    }
                }
            }
            coord[v] = median(std::move(values));
        }
    }
    // 共享预处理用消费者位置补齐，后续仍可迁移。
    for (auto it = g.op_topo.rbegin(); it != g.op_topo.rend(); ++it) {
        if (coord[*it] < 0) {
            std::vector<double> values;
            for (int u : g.ops[*it].succs) {
                if (coord[u] >= 0) {
                    values.push_back(coord[u]);
                }
            }
            coord[*it] = median(std::move(values));
        }
    }
    return coord;
}

// 一次搜索分支的全部状态，负责生成候选、完整复核、缓存和提交最优分组。

}
}
