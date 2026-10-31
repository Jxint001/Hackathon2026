#pragma once
#include "search/candidate.hpp"
namespace npu {
namespace search_detail {
struct Grid;
// 固定的结构摘要和候选变换；不评估、不更新搜索状态。
class CandidateGenerator {
  public:
    const Problem &problem;
    int cores, size, dominant = 0;
    std::vector<int> topo, depth, dfs, atom;
    std::vector<std::vector<int>> members;
    std::vector<std::array<double, NUM_PIPES>> loads;
    std::array<double, NUM_PIPES> total{};
    struct PlacementResources {
        double peak, ddr, total_peak;
    };
    CandidateGenerator(const Problem &g, int n);
    std::vector<int> core_assignment(const Sequences &seq) const;
    Sequences sequences_for_assignment(const std::vector<int> &assignment,
                                       const std::vector<int> &order) const;
    double resource_lower_bound(int active) const;
    std::vector<int> dependency_dfs_order() const;
    Grid recognize_grid() const;
    std::vector<int> coordinate_assignment(const std::vector<double> &coord, int active,
                                           bool multi_pipe) const;
    Sequences delay_producers(Sequences seq, double factor, bool expanding_only,
                              int selected_depth = -1) const;
    PlacementResources placement_resources(const std::vector<int> &assignment) const;
    double placement_cost(const std::vector<int> &assignment) const;
    Sequences window_order(const Sequences &reference, int window, double bias) const;
    Sequences pull_consumers(Sequences seq, int selected_depth) const;
};
}
}
