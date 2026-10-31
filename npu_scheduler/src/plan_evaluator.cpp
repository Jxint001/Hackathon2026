#include "plan_evaluator.hpp"
#include "evaluation/model.hpp"
namespace npu {
using namespace evaluation_detail;
// 顺序保持：任务构造 → 每核排序与内存展开 → 全局事件模拟。
PlanEvaluation evaluate_plan(const Problem &g, const std::vector<int> &mapping, const Plan &solution,
                             int cores, ExecutionTrace *trace) {
    PlanEvaluation result;
    try {
        EvaluationWorkspace workspace;
        build_tasks(g, mapping, solution, cores, workspace, result);
        auto &tasks = workspace.tasks;
        auto &links = workspace.links;
        auto &rank = workspace.rank;
        if (trace) {
            trace->sequences.assign(cores, {});
            trace->starts.assign(g.ops.size(), 0);
        }
        for (int c = 0; c < cores; ++c) {
            auto &task = tasks[c];
            local_order(task, rank);
            if (trace) {
                for (int u : task.sequence) {
                    if (!task.ops[u].copy) {
                        auto it = std::lower_bound(
                            g.ops.begin(), g.ops.end(), task.ops[u].id,
                            // 为轨迹映射二分查找原操作索引。
                            [](const Op &op, Key id) { return static_cast<Key>(op.id) < id; });
                        trace->sequences[c].push_back(static_cast<int>(it - g.ops.begin()));
                    }
                }
            }
            result.spill_bytes += insert_spills(task);
            prepare_memory(task);
        }
        result.copy_bytes += result.spill_bytes;
        std::vector<double> starts;
        result.makespan =
            global_replay(tasks, links, trace ? &starts : nullptr, trace ? &trace->core_ends : nullptr);
        result.valid = true;
        if (trace) {
            int base = 0;
            for (const auto &task : tasks) {
                for (int u = 0; u < static_cast<int>(task.ops.size()); ++u) {
                    if (!task.ops[u].copy) {
                        auto it = std::lower_bound(
                            g.ops.begin(), g.ops.end(), task.ops[u].id,
                            // 为开始时间回填二分查找原操作索引。
                            [](const Op &op, Key id) { return static_cast<Key>(op.id) < id; });
                        trace->starts[it - g.ops.begin()] = starts[base + u];
                    }
                }
                base += static_cast<int>(task.ops.size());
            }
        }
    } catch (const InvalidReview &) {
        result.valid = false;
    }
    return result;
}
}
