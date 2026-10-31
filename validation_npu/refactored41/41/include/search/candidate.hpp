#pragma once
#include "plan_evaluator.hpp"
namespace npu {
namespace search_detail {
using Sequences = std::vector<std::vector<int>>;
struct Candidate {
    Sequences seq;
    int chunk = 1;
    int family = 0;
};
struct SearchSeed {
    Candidate candidate;
    PlanEvaluation score;
    std::vector<int> assignment;
};
// 只作命名，不改变原阶段编号和顺序。
enum class SearchStage {
    DelayProducers = 1,
    CapacityTiles = 5,
    SpatialOverlap = 6,
    AlignedTiles = 7,
    ConvolutionOrder = 8,
    MatrixStarts = 9,
    ChainMoves = 10
};
}
}
