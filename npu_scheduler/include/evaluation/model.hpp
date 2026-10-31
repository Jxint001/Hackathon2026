#pragma once
#include "plan_evaluator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <queue>
#include <tuple>
namespace npu {
namespace evaluation_detail {
using Key = std::uint64_t;
constexpr std::array<int, 5> PIPE_ORDER = {2, 3, 0, 1, 4};
constexpr std::array<Id, 3> CAPACITY = {0, L1_CAPACITY, UB_CAPACITY};
constexpr double EPS = 1e-9;
// 对普通或 PMR 向量排序去重。
template <class T, class A> void distinct(std::vector<T, A> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}
// 在候选模拟内部中止非法、容量不可行或发生死锁的方案。
struct InvalidReview {};
// 使用调用方 PMR 内存资源分配的整数向量。
using Indices = std::pmr::vector<int>;
// 单核任务里的一个可模拟指令，包括计算、边界搬运及额外换出换入。
struct Instruction {
    Key id = 0;
    int pipe = 0, copy = 0, group = -1;
    double duration = 1;
    Indices inputs, outputs, preds, succs;
    // 使用指定内存资源初始化指令邻接与输入输出数组。
    explicit Instruction(std::pmr::memory_resource *memory)
        : inputs(memory), outputs(memory), preds(memory), succs(memory) {}
};
// 单核模拟中的一个具体张量存储实例，换入后可产生新的实例。
struct Buffer {
    Key id = 0;
    Id size = 0;
    int memory = 0;
    Indices writers, readers;
    // 初始化一个张量缓冲区及其读写者容器。
    Buffer(Key key, Id bytes, int location, std::pmr::memory_resource *resource)
        : id(key), size(bytes), memory(location), writers(resource), readers(resource) {}
};
// 一个核心上全部子图合并后的模拟任务，不为同核子图额外插入边界搬运。
struct Task {
    std::pmr::memory_resource *memory;
    std::vector<Instruction> ops;
    std::vector<Buffer> buffers;
    std::vector<int> sequence;
    std::array<std::vector<int>, 5> fifo;
    std::vector<std::pair<int, int>> memory_edges;
    // 创建空的单核任务。
    explicit Task(std::pmr::memory_resource *resource) : memory(resource) {}
};
// 跨核 COPY 的同步依赖边。
struct CrossLink {
    int source_core, source, target_core, target;
};

struct Running {
    int node = -1;
    double end = 0, work = 0;
    bool ddr = false;
};

// arena 必须先构造、最后析构，Task 内的 PMR 容器借用它。
struct EvaluationWorkspace {
    std::pmr::monotonic_buffer_resource arena;
    std::vector<Task> tasks;
    std::vector<CrossLink> links;
    std::vector<int> rank;
};
// 每次释放保留来源；再次分配必须依赖这些释放操作。
struct MemoryCredit {
    Id bytes;
    int tensor;
};
struct MemoryState {
    std::vector<int> refs;
    std::vector<char> resident;
    std::array<std::deque<MemoryCredit>, 3> credits;
    std::array<Id, 3> used{};
    std::size_t allocation_cursor = 0;
    explicit MemoryState(int tensors) : refs(tensors), resident(tensors) {}
};
// 每条管道只有一个运行项；剩余依赖与释放时间共同决定就绪。
struct SimulationState {
    std::vector<int> remaining;
    std::vector<double> release;
    std::vector<Running> running;
    std::vector<std::size_t> cursor;
    double now = 0, last = 0;
    int done = 0;
    SimulationState(int nodes, std::size_t pipes)
        : remaining(nodes), release(nodes), running(pipes), cursor(pipes) {}
};
int buffer(Task &task, Key id, Id bytes, int memory);
int copy(Task &task, Key id, int kind, int input, int output, int group);
void build_views(Task &task);
void local_order(Task &task, const std::vector<int> &group_rank);
double insert_spills(Task &task);
void advance(std::vector<Running> &running, double elapsed);
void project(std::vector<Running> &running, double now);
bool uses_ddr(const Task &task, int u);
void prepare_memory(Task &task);
double global_replay(const std::vector<Task> &tasks, const std::vector<CrossLink> &links,
                     std::vector<double> *starts, std::vector<double> *core_ends);
void build_tasks(const Problem &g, const std::vector<int> &mapping, const Plan &solution, int cores,
                 EvaluationWorkspace &workspace, PlanEvaluation &result);
}
}
