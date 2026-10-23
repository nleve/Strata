#pragma once

#include <charconv>
#include <climits>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace strata::program {

// Stage 0 stays on the caller; later stages keep a thread each so their thread-local copy streams are reused.
// One orchestration thread calls run(), which always waits for every stage, including when a stage throws.
class SnapshotWorkers {
public:
    explicit SnapshotWorkers(size_t stages) : failures_(stages) {
        if (stages == 0) throw std::invalid_argument("snapshot workers need at least one stage");
        try {
            for (size_t k = 1; k < stages; ++k) workers_.emplace_back([this, k] { loop(k); });
        } catch (...) {
            stop();                         // also join workers when thread creation fails partway through
            throw;
        }
    }
    ~SnapshotWorkers() { stop(); }
    SnapshotWorkers(const SnapshotWorkers&) = delete;
    SnapshotWorkers& operator=(const SnapshotWorkers&) = delete;

    void run(std::function<void(size_t)> work) {
        for (auto& failure : failures_) failure = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            work_ = std::move(work);         // any allocation for the callable happened before jobs are released
            remaining_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        call(0);
        {
            std::unique_lock<std::mutex> lock(mu_);
            done_.wait(lock, [&] { return remaining_ == 0; });
        }
        work_ = nullptr;
        for (const auto& failure : failures_) if (failure) std::rethrow_exception(failure);
    }

private:
    void call(size_t stage) noexcept {
        try { work_(stage); } catch (...) { failures_[stage] = std::current_exception(); }
    }
    void loop(size_t stage) {
        size_t seen = 0;
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || seen != generation_; });
            if (stopping_) return;
            seen = generation_;
            lock.unlock();
            call(stage);
            lock.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mu_); stopping_ = true; }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    std::mutex mu_;
    std::condition_variable ready_, done_;
    std::function<void(size_t)> work_;
    std::vector<std::exception_ptr> failures_;
    size_t generation_ = 0, remaining_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;       // last: threads only see fully initialized members
};

// A first CPU plus one consecutive CPU per copy worker, without accepting junk or overflowing the last CPU.
inline bool parse_snapshot_copy_cpu(std::string_view text, size_t workers, int& first) {
    if (text.empty() || workers == 0) return false;
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < 0 ||
        workers - 1 > size_t(INT_MAX - value)) return false;
    first = value;
    return true;
}

}  // namespace strata::program
