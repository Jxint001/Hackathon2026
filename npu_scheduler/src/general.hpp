#pragma once

#include "analysis.hpp"
#include "search.hpp"

namespace npu {

// 只提供分核、迁移和排序动作，不持有搜索状态。
ActionProvider general_actions(std::shared_ptr<const SchedulingGroups> groups, int cores,
                               std::size_t coverage);

} // npu 命名空间
