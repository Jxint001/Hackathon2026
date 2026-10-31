// 定义计算图、张量、操作、分组及固定硬件参数，声明图读取与分组接口。
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace npu {
struct Json;
constexpr std::int64_t L1_CAPACITY = 1572864, UB_CAPACITY = 131072;
constexpr double DDR_BANDWIDTH = 60.0, CROSS_CORE_DELAY = 1000.0;
constexpr int NUM_PIPES = 5;
// 输入图原始节点编号及张量字节数使用的有符号 64 位整数。
using Id = std::int64_t;
// 计算图中的一个原始操作，包含计算操作和 COPY 边界操作。
struct Op {
    // pipe：M、V、MTE2、MTE3、FIX；copy：计算、输入搬运、输出搬运。
    Id id = 0;
    int pipe = 0, copy = 0;
    double cycles = 0;
    std::vector<int> inputs, outputs, preds, succs;
};
// 计算图中的张量及其计算读写关系。
struct Tensor {
    // memory：0 DDR、1 L1、2 UB；size 单位为字节。
    Id id = 0, size = 0;
    int memory = 0;
    bool output = false;
    std::vector<int> producers, consumers;
};
// 一组作为同一子图提交的计算操作，也是初始解分核的基本单位。
struct Group {
    std::vector<int> ops, preds, succs, tensors;
    std::array<double, NUM_PIPES> load{};
    double work = 0, rank = 0;
};
// 输入图只在解析时填写，求解阶段以 const 引用使用。
struct Problem {
    std::vector<Op> ops;
    std::vector<Tensor> tensors;
    std::vector<int> op_topo;
};

// 非空分组统一由 make_partition 构造，返回时所有派生字段完整。
// groups 中的操作顺序由调用方决定，构造函数不重排。
struct Partition {
    std::vector<Group> groups;
    std::vector<int> op_group, topo;
};
// 读取图并建立保留 COPY 桥接关系的计算依赖。
Problem read_graph(const Json &root);
// 从操作列表一次性建立映射、依赖、负载和拓扑信息。
Partition make_partition(const Problem &problem, std::vector<std::vector<int>> group_ops);
// 构造满足共享写者约束的无环初始分组。
Partition build_groups(const Problem &graph);
// 表示无法生成通过内部合法性复核的方案。
struct NoPlan : std::runtime_error {
    using std::runtime_error::runtime_error;
};
}
