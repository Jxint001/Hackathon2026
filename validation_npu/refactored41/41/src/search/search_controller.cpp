#include "search/search_controller.hpp"
#include "search/common.hpp"
#include "thread_pool.hpp"
namespace npu {
namespace search_detail {
// 保存只读输入和当前最优方案，并准备三类职责：
// generator 分析依赖与共享写者、构造候选；evaluator 模拟候选；state 保存预算、缓存和结果。
// 预算由图规模决定，是每阶段的尝试次数上限，不是秒数；构造器本身不启动搜索。
SearchController::SearchController(const Problem &g, int n, const Solution &baseline)
    : problem(g), cores(n), size(static_cast<int>(g.ops.size())), generator(g, n), evaluator(g, n),
      state(baseline,
            std::max(120, std::min(900, 18000000 / std::max(1, static_cast<int>(g.op_topo.size()))))) {}

bool SearchController::better(const PlanEvaluation &r) const {
    return r.valid && (r.makespan < state.best.score.makespan - 0.5 ||
                       (std::abs(r.makespan - state.best.score.makespan) <= 0.5 &&
                        r.copy_bytes < state.best.score.copy_bytes - 0.5));
}

void SearchController::remember_seed(const Candidate &candidate, const PlanEvaluation &score) {
    if (!score.valid) {
        return;
    }
    const auto assignment = generator.core_assignment(candidate.seq);
    for (auto &seed : state.seeds) {
        if (seed.assignment == assignment && seed.candidate.family == candidate.family) {
            if (std::tie(score.makespan, score.copy_bytes) <
                std::tie(seed.score.makespan, seed.score.copy_bytes)) {
                seed = {candidate, score, assignment};
            }
            return;
        }
    }
    state.seeds.push_back({candidate, score, assignment});
}

// 并行只填结果槽，选优仍按候选原顺序，保证确定性。
void SearchController::prefetch(const std::vector<Candidate> &candidates) {
    state.prefetched_scores.clear();
    if (SEARCH_THREADS == 1 || generator.topo.size() < 256 || state.phase_attempts >= state.phase_limit) {
        return;
    }
    std::vector<std::pair<std::uint64_t, std::size_t>> pending;
    std::set<std::uint64_t> seen;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto &c = candidates[i];
        auto hash = evaluator.candidate_hash(c.seq, c.chunk);
        if (!state.seen_candidates.count(hash) && !state.evaluation_cache.count(hash) &&
            seen.insert(hash).second && competitive(c.seq)) {
            pending.emplace_back(hash, i);
        }
    }
    std::vector<PlanEvaluation> results(pending.size());
    // 在线程池中复核一个待处理候选。
    parallel_for(pending.size(), true,
                 [&](std::size_t i) { results[i] = evaluator.evaluate(candidates[pending[i].second]); });
    for (std::size_t i = 0; i < pending.size(); ++i) {
        state.prefetched_scores.emplace(pending[i].first, results[i]);
    }
}

// 普通候选的统一入口：检查预算与重复，筛掉资源代价已无竞争力的候选，再复用或计算成绩。
// 有效且满足时间优先、搬运量次优的比较规则时，才构造完整方案并替换当前最优。
// 更新方案后丢弃旧执行轨迹，让后续顺序调整读取新方案的实际模拟顺序。
// 尝试计数在去重之后、剪枝之前增加；缓存命中仍占原阶段预算。
PlanEvaluation SearchController::consider(const Sequences &seq, int chunk) {
    PlanEvaluation empty;
    if (state.phase_attempts >= state.phase_limit) {
        return empty;
    }
    auto hash = evaluator.candidate_hash(seq, chunk);
    if (!state.seen_candidates.insert(hash).second) {
        return empty;
    }
    ++state.phase_attempts;
    ++state.total_attempts;
    auto cached = state.evaluation_cache.find(hash);
    PlanEvaluation result;
    if (cached != state.evaluation_cache.end()) {
        result = cached->second;
    } else {
        if (!competitive(seq)) {
            state.evaluation_cache.emplace(hash, empty);
            return empty;
        }
        auto prepared = state.prefetched_scores.find(hash);
        result =
            prepared == state.prefetched_scores.end() ? evaluator.evaluate({seq, chunk}) : prepared->second;
        state.evaluation_cache.emplace(hash, result);
    }
    if (better(result)) {
        auto s = evaluator.materialize_solution(seq, chunk);
        state.best = std::move(s);
        state.best.score.makespan = result.makespan;
        state.best.score.copy_bytes = result.copy_bytes;
        state.best_trace.reset();
        ++state.best_revision;
    }
    return result;
}

void SearchController::consider_batch(const std::vector<Candidate> &candidates) {
    for (std::size_t i = 0; i < candidates.size() && state.phase_attempts < state.phase_limit; i += 10) {
        std::vector<Candidate> batch(candidates.begin() + i,
                                     candidates.begin() + std::min(i + 10, candidates.size()));
        prefetch(batch);
        for (const auto &c : batch) {
            const auto score = consider(c.seq, c.chunk);
            if (state.collect_seeds) {
                remember_seed(c, score);
            }
        }
    }
}

const ExecutionTrace &SearchController::trace_best() {
    if (!state.best_trace) {
        ExecutionTrace trace;
        if (!evaluate_plan(problem, state.best.partition.op_group, state.best.plan, cores, &trace).valid) {
            throw NoPlan("selected candidate cannot be replayed");
        }
        state.best_trace = std::move(trace);
    }
    return *state.best_trace;
}

bool SearchController::competitive(const Sequences &seq) const {
    const auto resources = generator.placement_resources(generator.core_assignment(seq));
    return std::max(resources.peak, resources.ddr) <=
           state.best.score.makespan + std::max(0.5, state.best.score.makespan * 1e-12);
}

// 剪枝失败依赖当时的最优值，不能作为另一搜索起点的无效结论。
void SearchController::copy_completed_to(SearchController &target) const {

    for (const auto &entry : state.evaluation_cache) {
        if (entry.second.valid) {
            target.state.evaluation_cache[entry.first] = entry.second;
        }
    }
}

// 先判断当前方案是否已接近资源下界，接近时跳过这一轮搜索。
// 能识别行列网格时，尝试让相邻计算共同分核以复用输入；识别失败只跳过网格候选。
// 随后尝试不同子图大小，比较分组对核内执行顺序的影响。
void SearchController::grid_candidates() {
    if (state.best.score.makespan <= generator.resource_lower_bound(cores) * 1.01) {
        return;
    }
    auto grid = generator.recognize_grid();
    if (grid.rows) {
        grid_search(grid);
    }
    granularity();
}

// 每阶段只负责一种候选搜索，始终从当前最优方案和已有结构信息出发。
// 开始时重置本阶段尝试数、去重集合和预评估结果；最优解与完整评估缓存跨阶段保留。
// 这里负责分派，不直接决定候选好坏；普通候选由 consider() 处理，多起点另有提升门槛。
void SearchController::phase(SearchStage step) {

    state.phase_attempts = 0;
    state.seen_candidates.clear();
    state.prefetched_scores.clear();
    if (step == SearchStage::DelayProducers) {
        // 让生产更靠近消费，减少过早生成的数据占用。
        producer_search();
    } else if (step == SearchStage::CapacityTiles) {
        // 按存储容量选择块尺寸，比较复用收益与换出代价。
        capacity_tiles();
    } else if (step == SearchStage::SpatialOverlap) {
        // 按空间关系尝试分核和顺序，兼顾计算量与管道重叠。
        overlap_search();
    } else if (step == SearchStage::AlignedTiles) {
        // 对齐主体条带、分摊尾块，并尝试复用相邻条带的输入。
        aligned_tiles();
    } else if (step == SearchStage::ConvolutionOrder) {
        // 保持分核，比较依赖层提前、消费者前移和生产者延后。
        convolution_orders();
    } else if (step == SearchStage::MatrixStarts) {
        // 保留若干矩阵分块起点，各自继续调整后再比较。
        matrix_starts();
    } else if (step == SearchStage::ChainMoves) {
        // 迁移整链或部分链，尝试减少核心负载不均，同时考虑新增通信。
        chain_moves();
    } else {
        throw NoPlan("unknown refinement step");
    }
}

// 完整搜索安排：先试网格与粒度，再依次改进数据驻留、分核布局和链的归属。
// 接着集中调整当前最优方案的核内顺序，最后对少量不同空间起点分别搜索。
// 这不是两个互斥的 Matmul/Conv 大分支：各阶段自行判断结构是否适用，结果进入同一最优状态。
// 顺序固定；枚举哪些块尺寸、边界和起点是各策略内部细节，不在这里展开。
Solution SearchController::run() {
    grid_candidates();
    for (SearchStage step :
         {SearchStage::DelayProducers, SearchStage::CapacityTiles, SearchStage::SpatialOverlap,
          SearchStage::AlignedTiles, SearchStage::ChainMoves, SearchStage::MatrixStarts}) {
        phase(step);
    }
    // 记住顺序优化前的起点，后续多起点搜索避免重复从同一序列出发。
    const auto starting_hash = state.seeds.empty() ? 0 : evaluator.candidate_hash(trace_best().sequences, 1);
    // 布局调整之后再改善核内顺序，避免只分好核却留下过多等待或换出。
    phase(SearchStage::ConvolutionOrder);
    // 换少量不同起点继续尝试，已有最好方案单独保留。
    multistart_search(starting_hash);
    return finish();
}

// 最优方案的分组信息已在构造时补齐，这里只累计统计并返回。
Solution SearchController::finish() {
    state.best.statistics.evaluations += state.total_attempts;
    return state.best;
}

}
}
