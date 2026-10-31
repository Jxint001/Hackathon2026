// 实现进程内复用的固定线程池；主线程参与工作，总线程数不超过 10。
#pragma once
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace npu {

#ifndef NPU_SEARCH_THREADS
#define NPU_SEARCH_THREADS 10
#endif
constexpr int SEARCH_THREADS = NPU_SEARCH_THREADS;
static_assert(SEARCH_THREADS >= 1 && SEARCH_THREADS <= 10, "thread limit is 10");

// 进程内复用的批量索引任务执行器，主线程与最多 9 个后台线程共同工作。
class ThreadPool {
    std::vector<std::thread> workers;
    std::mutex mutex, invocation;
    std::condition_variable wake, finished;
    std::function<void(std::size_t)> job;
    std::vector<std::exception_ptr> errors;
    std::atomic<std::size_t> next{0};
    std::size_t count = 0, generation = 0, completed = 0;
    bool stopping = false;

    // 从原子索引领取并执行任务，按任务索引保存异常。
    void drain() noexcept {
        for (;;) {
            auto i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) {
                return;
            }
            try {
                job(i);
            } catch (...) {
                errors[i] = std::current_exception();
            }
        }
    }
    // 通知工作线程退出并等待全部线程结束。
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        for (auto &worker : workers) {
            worker.join();
        }
    }

  public:
    // 创建固定数量工作线程，调用方主线程也参与工作。
    ThreadPool() {
        try {
            // 工作线程等待新批次、领取任务并报告完成。
            for (int i = 1; i < SEARCH_THREADS; ++i) {
                workers.emplace_back([this] {
                    std::size_t observed = 0;
                    for (;;) {
                        std::unique_lock<std::mutex> lock(mutex);
                        // 判断工作线程是否可以从等待中唤醒。
                        wake.wait(lock, [&] { return stopping || generation != observed; });
                        if (stopping) {
                            return;
                        }
                        observed = generation;
                        lock.unlock();
                        drain();
                        lock.lock();
                        ++completed;
                        finished.notify_one();
                    }
                });
            }
        } catch (...) {
            stop();
            throw;
        }
    }
    // 析构线程池并回收所有工作线程。
    ~ThreadPool() {
        stop();
    }
    // 禁止复制线程池，避免复制线程和同步状态。
    ThreadPool(const ThreadPool &) = delete;
    // 禁止复制赋值线程池。
    ThreadPool &operator=(const ThreadPool &) = delete;

    // 发布一批索引任务，主线程参与执行并等待所有任务完成。
    void run(std::size_t n, std::function<void(std::size_t)> function) {
        std::lock_guard<std::mutex> call(invocation);
        {
            std::lock_guard<std::mutex> lock(mutex);
            errors.assign(n, {});
            job = std::move(function);
            count = n;
            next = 0;
            completed = 0;
            ++generation;
        }
        wake.notify_all();
        drain();
        std::unique_lock<std::mutex> lock(mutex);
        // 判断本批次全部后台线程是否执行完毕。
        finished.wait(lock, [&] { return completed == workers.size(); });
        job = {};
        for (const auto &error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }
};

// 取得进程内唯一的复用线程池。
inline ThreadPool &thread_pool() {
    static ThreadPool pool;
    return pool;
}
// 根据规模选择串行或固定线程池执行索引任务。
template <class F> void parallel_for(std::size_t count, bool worthwhile, F function) {
    if (SEARCH_THREADS == 1 || !worthwhile || count < 2) {
        for (std::size_t i = 0; i < count; ++i) {
            function(i);
        }
    } else {

        thread_pool().run(count, std::move(function));
    }
}
}
