// 将最终分组及各核顺序序列化为提交契约规定的两个 JSON 字段。
#include "plan.hpp"
#include <sstream>

namespace npu {
// 按提交格式写出操作归属和各核子图顺序。
std::string serialize_plan(const Problem &g, const Solution &plan, int cores) {
    std::ostringstream out;
    out << "{\n  \"node_to_subgraph\": {";
    bool first = true;
    for (int u = 0; u < static_cast<int>(g.ops.size()); ++u) {
        if (!g.ops[u].copy) {
            if (!first) {
                out << ',';
            }
            first = false;
            out << "\n    \"" << g.ops[u].id << "\": " << plan.partition.op_group[u];
        }
    }
    out << "\n  },\n  \"core_schedules\": [";
    for (int c = 0; c < cores; ++c) {
        if (c) {
            out << ", ";
        }
        out << '[';
        bool first_group = true;
        for (int u : plan.plan.order) {
            if (plan.plan.core.at(u) == c) {
                if (!first_group) {
                    out << ", ";
                }
                first_group = false;
                out << u;
            }
        }
        out << ']';
    }
    out << "]\n}\n";
    return out.str();
}
}
