#include "strata/program/snapshot_workers.hpp"

#include <chrono>
#include <cstdio>
#include <future>
#include <latch>
#include <string>

int main() {
    using strata::program::SnapshotWorkers;
    using strata::program::parse_snapshot_copy_cpu;
    using namespace std::chrono_literals;
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    };
    int cpu = -1;
    check(parse_snapshot_copy_cpu("17", 2, cpu) && cpu == 17, "parse first worker CPU");
    for (const char* bad : {"", "-1", "1x", "2147483648"})
        check(!parse_snapshot_copy_cpu(bad, 2, cpu), "reject malformed or overflowing CPU");
    check(!parse_snapshot_copy_cpu(std::to_string(INT_MAX), 2, cpu), "reject overflow in the last worker CPU");

    // A two-GPU split needs one background worker. The latch requires overlap, and both threads persist.
    SnapshotWorkers pair(2);
    std::vector<std::thread::id> before(2), after(2);
    for (int round = 0; round < 2; ++round) {
        std::latch started(2);
        pair.run([&](size_t stage) {
            after[stage] = std::this_thread::get_id();
            started.count_down();
            started.wait();
        });
        check(after[0] == std::this_thread::get_id(), "stage zero stays on the caller");
        if (round) check(after == before, "workers persist across rounds");
        before = after;
    }

    // A failure on the caller or a worker must wait for every outstanding stage before run() throws.
    struct StageFailure {};
    SnapshotWorkers workers(3);
    for (size_t failing : {size_t(0), size_t(1)}) {
        std::promise<void> blocked, failed, release;
        auto unblocked = release.get_future();
        auto result = std::async(std::launch::async, [&] {
            workers.run([&](size_t stage) {
                if (stage == failing) { failed.set_value(); throw StageFailure{}; }
                if (stage == 2) { blocked.set_value(); unblocked.wait(); }
            });
        });
        blocked.get_future().wait();
        failed.get_future().wait();
        check(result.wait_for(50ms) == std::future_status::timeout, "failed run waits for its outstanding stage");
        release.set_value();
        try { result.get(); check(false, "stage failure propagates after the join"); }
        catch (const StageFailure&) {}
    }
    workers.run([](size_t) {});               // the next successful round must not inherit a failure
    if (!failures) std::puts("snapshot_workers_test OK");
    return failures ? 1 : 0;
}
