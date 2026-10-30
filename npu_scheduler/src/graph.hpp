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
// Step2 starts spilling at 90%; execution still uses the full hard capacity.
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
    int storage = -1; // DDR=-1, L1=0, UB=1.
    Cost bytes = 0;
    std::vector<int> producers, consumers;
};

struct Operation {
    Id id = 0;
    std::string name;
    int pipe = 0; // MTE2, MTE3, M, V, FIX.
    Cost cycles = 0;
    int copy = 0; // 0: ordinary, 1: COPY_IN, 2: COPY_OUT.
    std::vector<int> inputs, outputs;
};

struct Graph {
    std::vector<Tensor> tensors;
    std::vector<Operation> ops;
    std::vector<std::pair<int, int>> direct;
    std::vector<Cost> direct_bytes; // Parallel to direct; absent data_size is zero.
    std::vector<int> work;
    std::vector<std::vector<int>> pred, succ; // Dependencies between non-COPY ops.
    std::vector<int> topo;
    Id max_id = 0;
};

struct Plan {
    std::vector<std::vector<int>> orders; // Original operation indices, one sgid per op.
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

} // namespace npu
