#include "blocks.hpp"

#include <functional>
#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace npu {
namespace {

std::vector<int> plan_ranks(const Graph& g, const Plan& plan) {
    auto next = g.succ;
    std::vector<int> degree(g.ops.size()), rank(g.ops.size());
    for (const auto& order : plan.orders)
        for (std::size_t i = 1; i < order.size(); ++i) next[order[i - 1]].push_back(order[i]);
    for (int v : g.work) for (int u : next[v]) ++degree[u];
    std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
    for (int v : g.work) if (!degree[v]) ready.push(v);
    int position = 0;
    while (!ready.empty()) {
        const int v = ready.top(); ready.pop(); rank[v] = position++;
        for (int u : next[v]) if (!--degree[u]) ready.push(u);
    }
    return rank;
}

} // 匿名命名空间

Plan emit_blocks(const CoreBlocks& blocks, int width, int steps) {
    Plan plan; plan.orders.resize(blocks.size());
    width = std::max(1, width); steps = std::max(1, steps);
    for (int c = 0; c < static_cast<int>(blocks.size()); ++c) {
        for (int begin = 0; begin < static_cast<int>(blocks[c].size()); begin += width) {
            const int end = std::min(begin + width, static_cast<int>(blocks[c].size()));
            int depth = 0;
            for (int b = begin; b < end; ++b) depth = std::max(depth, static_cast<int>(blocks[c][b].stages.size()));
            // 每批交错推进 steps 层，再推进下一批层。
            for (int k = 0; k < depth;) {
                const int until = k + std::min(steps, depth - k);
                for (int b = begin; b < end; ++b)
                    for (int d = k; d < std::min(until, static_cast<int>(blocks[c][b].stages.size())); ++d) {
                        const auto& stage = blocks[c][b].stages[d];
                        plan.orders[c].insert(plan.orders[c].end(), stage.begin(), stage.end());
                    }
                k = until;
            }
        }
    }
    return plan;
}

Plan place_blocks(const Graph& g, const Plan& base, const CoreBlocks& blocks,
                  int width, int steps, bool late) {
    const int n = static_cast<int>(g.ops.size());
    std::vector<int> owner(n, -1), degree(n);
    const auto rank = plan_ranks(g, base);
    std::vector<std::pair<int, int>> priority(n);
    for (int c = 0; c < static_cast<int>(base.orders.size()); ++c)
        for (int i = 0; i < static_cast<int>(base.orders[c].size()); ++i) {
            const int v = base.orders[c][i]; owner[v] = c;
            priority[v] = {rank[v], 0};
        }
    // 每批以最早或最晚的原位置为锚点，未选中的操作保留原优先级。
    for (int c = 0; c < static_cast<int>(blocks.size()); ++c) {
        for (int begin = 0; begin < static_cast<int>(blocks[c].size()); begin += std::max(1, width)) {
            CoreBlocks part(1);
            const int end = std::min(begin + std::max(1, width), static_cast<int>(blocks[c].size()));
            part[0].insert(part[0].end(), blocks[c].begin() + begin, blocks[c].begin() + end);
            const auto emitted = emit_blocks(part, width, steps);
            if (emitted.orders[0].empty()) continue;
            int anchor = late ? 0 : INT32_MAX;
            for (int v : emitted.orders[0]) anchor = late ? std::max(anchor, rank[v]) : std::min(anchor, rank[v]);
            int position = 0;
            for (int v : emitted.orders[0]) priority[v] = {anchor, ++position};
        }
    }
    using Entry = std::tuple<int, int, int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> ready;
    const auto offer = [&](int v) { ready.emplace(priority[v].first, priority[v].second, owner[v], v); };
    for (int v : g.work) { degree[v] = static_cast<int>(g.pred[v].size()); if (!degree[v]) offer(v); }
    Plan result; result.orders.resize(base.orders.size());
    while (!ready.empty()) {
        const int v = std::get<3>(ready.top()); ready.pop();
        result.orders[owner[v]].push_back(v);
        for (int u : g.succ[v]) if (!--degree[u]) offer(u);
    }
    return result;
}

std::vector<int> spill_targets(const Graph& g, const Plan& plan, const Evaluation& score, int limit) {
    std::vector<int> owner(g.ops.size(), -1), rank(g.ops.size());
    for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c)
        for (int i = 0; i < static_cast<int>(plan.orders[c].size()); ++i) {
            const int v = plan.orders[c][i]; owner[v] = c; rank[v] = i;
        }
    std::vector<std::pair<double, int>> ranked;
    for (int t = 0; t < static_cast<int>(g.tensors.size()); ++t) {
        const auto& tensor = g.tensors[t];
        if (tensor.storage < 0 || tensor.bytes == 0) continue;
        double spread = 0;
        for (int c = 0; c < static_cast<int>(plan.orders.size()); ++c) {
            int first = INT32_MAX, last = -1, count = 0;
            for (int v : tensor.consumers) if (owner[v] == c) {
                first = std::min(first, rank[v]); last = std::max(last, rank[v]); ++count;
            }
            if (count > 1) spread += static_cast<double>(last - first + 1 - count) / count;
        }
        if (spread <= 0) continue;
        const Cost spill = score.tensor_spill_bytes.empty() ? 0 : score.tensor_spill_bytes[t];
        ranked.emplace_back(-static_cast<double>(spill) - tensor.bytes * std::min(1.0, spread / 128), t);
    }
    std::sort(ranked.begin(), ranked.end());
    std::vector<int> result;
    for (const auto& entry : ranked) {
        result.push_back(entry.second);
        if (static_cast<int>(result.size()) >= limit) break;
    }
    return result;
}

CoreBlocks consumer_blocks(const Graph& g, const GraphAnalysis& facts,
                           const Plan& base, const TensorBatch& config) {
    const int n = static_cast<int>(g.ops.size()), cores = static_cast<int>(base.orders.size());
    std::vector<int> owner(n, -1), rank(n), assigned(n);
    for (int c = 0; c < cores; ++c)
        for (int i = 0; i < static_cast<int>(base.orders[c].size()); ++i) {
            const int v = base.orders[c][i]; owner[v] = c; rank[v] = i;
        }
    CoreBlocks result(cores);
    for (int t : config.tensors) for (int c = 0; c < cores; ++c) {
        std::vector<int> users;
        for (int v : g.tensors[t].consumers) if (owner[v] == c && !assigned[v]) users.push_back(v);
        std::sort(users.begin(), users.end(), [&](int a, int b) { return rank[a] < rank[b]; });
        for (int start = 0; start < static_cast<int>(users.size()); start += std::max(1, config.batch)) {
            const int end = std::min(start + std::max(1, config.batch), static_cast<int>(users.size()));
            if (end - start < 2) continue;
            const int anchor = rank[users[start]];
            std::vector<char> chosen(n);
            std::vector<int> sequence;
            int discovered = 0;
            // 仅补齐锚点之后、同核尚未完成的前驱，限制闭包规模。
            std::vector<std::pair<int, bool>> stack;
            for (int i = end - 1; i >= start; --i) stack.emplace_back(users[i], false);
            while (!stack.empty() && discovered <= config.cone_limit) {
                const auto item = stack.back(); stack.pop_back();
                const int v = item.first;
                if (item.second) { sequence.push_back(v); continue; }
                if (chosen[v] || assigned[v] || owner[v] != c || rank[v] < anchor) continue;
                chosen[v] = 1; ++discovered;
                stack.emplace_back(v, true);
                for (auto it = g.pred[v].rbegin(); it != g.pred[v].rend(); ++it) stack.emplace_back(*it, false);
            }
            if (discovered > config.cone_limit || sequence.size() < 2) continue;
            ComputeBlock block;
            if (config.steps <= 0) {
                for (int v : sequence) block.stages.push_back({v});
            } else {
                int lo = INT32_MAX, hi = 0;
                for (int v : sequence) { lo = std::min(lo, facts.depth[v]); hi = std::max(hi, facts.depth[v]); }
                block.stages.resize(hi - lo + 1);
                for (int v : sequence) block.stages[facts.depth[v] - lo].push_back(v);
            }
            for (int v : sequence) assigned[v] = 1;
            result[c].push_back(std::move(block));
        }
    }
    return result;
}

Plan pack_consumers(const Graph& graph, const GraphAnalysis& facts,
                    const Plan& base, const TensorBatch& config) {
    const auto blocks = consumer_blocks(graph, facts, base, config);
    return place_blocks(graph, base, blocks, config.width, config.steps <= 0 ? INT32_MAX : config.steps, config.late);
}

Plan move_block(const Graph& g, const GraphAnalysis& facts, const Plan& base,
                const ComputeBlock& block, int destination) {
    std::vector<int> owner(g.ops.size(), -1);
    for (int c = 0; c < static_cast<int>(base.orders.size()); ++c)
        for (int v : base.orders[c]) owner[v] = c;
    for (const auto& stage : block.stages) for (int v : stage)
        for (int u : facts.co_groups[facts.co_group[v]]) owner[u] = destination;
    const auto rank = plan_ranks(g, base);
    auto order = g.work;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return rank[a] < rank[b]; });
    Plan result; result.orders.resize(base.orders.size());
    for (int v : order) result.orders[owner[v]].push_back(v);
    return result;
}

} // npu 命名空间
