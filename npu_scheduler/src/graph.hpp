#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace npu {

using Id = std::uint64_t;
using Cost = std::int64_t;
inline constexpr std::array<Cost, 2> kCapacity{{1572864, 131072}};
// 占用超过 90% 时尝试换出，硬上限仍为完整容量。
inline constexpr std::array<Cost, 2> kSpillThreshold{{kCapacity[0] * 90 / 100, kCapacity[1] * 90 / 100}};
inline constexpr Cost kBandwidth = 60;
inline constexpr Cost kCrossCoreDelay = 1000;
inline constexpr int kPipeCount = 5;
inline constexpr Cost kInfinity = INT64_MAX / 8;

inline Cost transfer_cycles(Cost bytes) {
    return bytes == 0 ? 1 : bytes / kBandwidth + (bytes % kBandwidth != 0);
}

struct Tensor {
    Id id = 0;
    int storage = -1; // 存储位置：DDR=-1，L1=0，UB=1。
    Cost bytes = 0;
    std::vector<int> producers, consumers;
};

struct Operation {
    Id id = 0;
    std::string name;
    int pipe = 0; // 流水线编号依次为 MTE2、MTE3、M、V、FIX。
    Cost cycles = 0;
    int copy = 0; // 0：普通操作，1：搬入，2：搬出。
    std::vector<int> inputs, outputs;
};

struct Graph {
    std::vector<Tensor> tensors;
    std::vector<Operation> ops;
    std::vector<std::pair<int, int>> direct;
    std::vector<Cost> direct_bytes; // 直接边的数据量，未提供时为 0。
    std::vector<int> work;
    std::vector<std::vector<int>> pred, succ; // 非 COPY 操作间的依赖。
    std::vector<int> topo;
    Id max_id = 0;
};

struct Plan {
    std::vector<std::vector<int>> orders; // 按核保存原图操作下标，每个操作单独成子图。
};

Graph read_graph(const std::filesystem::path& path);
bool valid_plan(const Graph& graph, const Plan& plan);
void write_plan(const Graph& graph, const Plan& plan,
                const std::filesystem::path& output);
std::vector<int> dependency_order(const Graph& graph, bool depth_first);

template<class T> void unique_sort(std::vector<T>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

} // npu 命名空间
