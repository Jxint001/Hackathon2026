#include "thread_pool.hpp"
#include "search/common.hpp"
#include "search/search_controller.hpp"
namespace npu {
namespace search_detail {
void SearchController::producer_search() {
    for (int round = 0; round < 2; ++round) {
        const double before = state.best.score.makespan;
        const auto seed = trace_best().sequences;
        std::vector<Candidate> batch;
        for (double factor : {1.0, 0.5, 2.0}) {
            batch.push_back({generator.delay_producers(seed, factor, false)});
        }
        batch.push_back({generator.delay_producers(seed, 1, true)});
        consider_batch(batch);
        if (state.best.score.makespan >= before - 0.5) {
            break;
        }
    }
}

void SearchController::convolution_orders() {
    if (!generator.total[0] ||
        std::accumulate(generator.total.begin() + 1, generator.total.end(), 0.0) == 0) {
        return;
    }
    std::vector<int> layers;
    for (int u : generator.topo) {
        layers.push_back(generator.depth[u]);
    }
    distinct(layers);
    if (layers.size() > 128) {
        return;
    }
    for (int round = 0; round < 6; ++round) {
        const double before = state.best.score.makespan;
        std::size_t prepared_revision = std::numeric_limits<std::size_t>::max(), prepared_until = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            int boundary = layers[i];
            auto seed = trace_best().sequences;
            // 把不超过指定依赖层的操作稳定前移。
            auto stage = [&](int layer) {
                auto seq = seed;
                // 识别应属于提前前缀的操作。
                for (auto &part : seq) {
                    std::stable_partition(part.begin(), part.end(),
                                          [&](int u) { return generator.depth[u] <= layer; });
                }
                return seq;
            };

            if (SEARCH_THREADS > 1 && generator.topo.size() >= 256 &&
                (state.best_revision != prepared_revision || i >= prepared_until)) {
                std::vector<Candidate> batch;
                prepared_until = std::min(i + 3, layers.size());
                for (std::size_t j = i; j < prepared_until; ++j) {
                    batch.push_back({stage(layers[j])});
                    batch.push_back({generator.pull_consumers(seed, layers[j])});
                    batch.push_back({generator.delay_producers(seed, 0.0, false, layers[j])});
                }
                prefetch(batch);
                prepared_revision = state.best_revision;
            }
            consider(stage(boundary));
            seed = trace_best().sequences;
            std::vector<Candidate> moves{{generator.pull_consumers(seed, boundary)},
                                         {generator.delay_producers(seed, 0.0, false, boundary)}};
            if (state.best_revision != prepared_revision) {
                prefetch(moves);
                prepared_revision = state.best_revision;
                prepared_until = i + 1;
            }
            for (const auto &move : moves) {
                consider(move.seq);
            }
            if (state.phase_attempts >= state.phase_limit) {
                return;
            }
        }
        if (state.best.score.makespan >= before - 0.5) {
            break;
        }
    }
}

}
}
