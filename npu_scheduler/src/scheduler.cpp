#include "scheduler.hpp"
#include "initial_plan.hpp"
#include "plan_search.hpp"

namespace npu {
/*
 * 整体方案：先整理计算组，再生成初始方案，最后逐步改进分核和执行顺序。
 * 输入是一张操作与张量组成的依赖图。先把共同写一个张量的操作合并，保证它们同核；
 * 若合并造成组间依赖成环，就继续合并相关组，再适当合并短链，减少初始化处理的单位。
 * 此时只决定哪些操作成组，还没有分核，也没有确定最终方案。
 *
 * 初始化尝试不同的核心使用数量、操作遍历顺序和分核方式。快速估算用于比较候选、
 * 提供后续排序的备选起点；去重后的分核候选仍需完整模拟，只有通过复核的方案才能入选。
 * 再尝试调整组的先后顺序，从这些方案中选择一个起点。
 *
 * 后续搜索关注计算是否均衡、输入能否复用、数据是否过早占用存储，以及流水线是否等待。
 * 对能识别出行列结构的计算尝试分块和遍历；对其他适用结构尝试空间分区、推迟生产、
 * 提前消费等调整。还会迁移部分计算，保留少量不同起点分别改进，减少只沿一个方案搜索的局限。
 *
 * 候选先检查是否值得继续评估，再通过完整模拟比较完成时间和搬运量。满足接受条件才更新
 * 最优方案。这里返回内存中的完整方案，由 main 转换为提交要求的操作分组和各核执行列表。
 */
Solution solve(const Problem &problem, int cores) {
    const auto partition = build_groups(problem); // 生成初始分组：把一些操作合成一组，作为初始分核的单位。
    if (partition.groups.empty()) {
        return {};
    }
    // 第二步：比较多种分核与分组执行顺序，选出通过内部模拟复核的初始方案。
    const auto baseline = generate_initial_plan(problem, partition, cores);
    // 第三步：从初始方案继续改进；各策略不适用时会跳过，已有最优方案始终保留。
    return improve_plan(problem, cores, baseline);
}
}
