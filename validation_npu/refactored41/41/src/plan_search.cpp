#include "plan_search.hpp"
#include "search/search_controller.hpp"
namespace npu {
// 接收已复核的初始方案，开始联合改进分组、分核和核内顺序。
// 控制器准备结构信息、评估工具和独立搜索状态，再由 run() 按固定顺序尝试各阶段。
// 各阶段共享当前最优方案；完整模拟负责比较候选，具体枚举参数留在各策略实现中。
// 返回完整最优方案，输入图始终只读。这里是完整搜索入口，不按用例名称选择算法。
Solution improve_plan(const Problem &graph, int cores, const Solution &baseline) {
    return search_detail::SearchController(graph, cores, baseline).run();
}
}
