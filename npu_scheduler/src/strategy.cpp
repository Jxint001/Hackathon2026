#include "strategy.hpp"
#include "analysis.hpp"
#include "general.hpp"
#include "matmul.hpp"
#include "search.hpp"

#include <future>

namespace npu {

std::vector<Plan> solve_core_limits(const Graph& graph, int cores) {
    const GraphAnalysis analysis(graph);
    const auto grids = find_reuse_grids(graph, analysis);
    std::size_t covered = 0;
    for (const auto& grid : grids) covered += grid.op_count();

    // 所有策略共用候选预算；规则网格保留较多分块候选。
    const std::size_t size = std::max<std::size_t>(1, graph.work.size());
    const bool mostly_grid = covered * 4 >= graph.work.size() * 3;
    const int budget = static_cast<int>(mostly_grid ?
        std::min<std::size_t>(640, std::max<std::size_t>(24, 9000000 / size)) :
        2 * std::min<std::size_t>(144, std::max<std::size_t>(20, 1800000 / size)) + 20);
    const auto compact = std::make_shared<const SchedulingGroups>(graph, analysis, true);
    auto broad_groups = std::make_shared<SchedulingGroups>(*compact);
    broad_groups->prepare_order(false);
    const std::shared_ptr<const SchedulingGroups> broad = std::move(broad_groups);
    const std::size_t ordinary = graph.work.size() - covered;
    const auto search_core_count = [&](int active) {
        std::vector<ActionProvider> providers;
        providers.push_back(general_actions(compact, active, (ordinary + 1) / 2));
        providers.push_back(general_actions(broad, active, ordinary / 2));
        for (const auto& grid : grids) providers.push_back(grid_actions(graph, analysis, grid, active));
        return search_actions(graph, analysis, *broad, active, budget, providers);
    };
    // 各核数独立并行搜索，只共享只读分析；最多四个工作线程。
    std::vector<std::future<Candidate>> searches;
    for (int active = 2; active <= cores; ++active)
        searches.push_back(std::async(std::launch::async, search_core_count, active));
    std::vector<Plan> results(cores + 1);
    Candidate best;
    // 按核数固定顺序取结果，不受线程完成顺序影响。
    for (int active = 2; active <= cores; ++active) {
        auto candidate = searches[active - 2].get();
        if (candidate.score.valid && (!best.score.valid || better(candidate, best))) best = std::move(candidate);
        if (best.score.valid) {
            results[active] = best.plan;
            results[active].orders.resize(active); // 多余核留空，不改变原方案。
        }
    }
    return results;
}

Plan solve(const Graph& graph, int cores) {
    auto plans = solve_core_limits(graph, cores);
    return std::move(plans[cores]);
}

} // npu 命名空间
