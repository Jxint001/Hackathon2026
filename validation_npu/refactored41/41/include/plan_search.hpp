// 声明由输入图结构驱动的完整搜索接口。
#pragma once
#include "plan_evaluator.hpp"

namespace npu {

// 从初始解执行全部搜索阶段，按图结构启用相应策略。
Solution improve_plan(const Problem &graph, int cores, const Solution &baseline);

}
