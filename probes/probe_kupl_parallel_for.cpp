#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <sched.h>
#include <vector>

#include "kupl.h"

namespace {

constexpr int kMaxWorkers = 256;
constexpr int kSamples = 101;

struct WorkArgs {
    int work = 0;
    alignas(64) unsigned long long checksum[kMaxWorkers] = {};
    alignas(64) int cpu[kMaxWorkers] = {};
};

double now_us()
{
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline unsigned long long do_page_work(size_t page, int work)
{
    unsigned long long value = page + 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < work; ++i) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
    }
    return value;
}

void kupl_loop(kupl_nd_range_t *range, void *opaque, int tid, int)
{
    auto *args = static_cast<WorkArgs *>(opaque);
    unsigned long long sum = args->checksum[tid];
    const auto &r = range->nd_range[0];
    for (size_t page = r.lower; page < r.upper; page += r.step) {
        sum += do_page_work(page, args->work);
    }
    args->checksum[tid] = sum;
    args->cpu[tid] = sched_getcpu();
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void run_case(size_t pages, int work, int workers, kupl_egroup_h group)
{
    WorkArgs kupl_args;
    kupl_args.work = work;

    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, pages);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = group,
        .concurrency = workers,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };

    std::vector<double> kupl_times;
    kupl_times.reserve(kSamples);
    for (int sample = -10; sample < kSamples; ++sample) {
        const double begin = now_us();
        kupl_parallel_for(&desc, kupl_loop, &kupl_args);
        const double kupl_us = now_us() - begin;
        if (sample >= 0) {
            kupl_times.push_back(kupl_us);
        }
    }

    std::printf(
        "KUPL pages=%zu work=%d: %.2f us\n",
        pages, work, median(kupl_times));
    std::printf("  CPUs:");
    for (int i = 0; i < workers; ++i) std::printf(" %d", kupl_args.cpu[i]);
    std::putchar('\n');

    const auto total = std::accumulate(
        kupl_args.checksum, kupl_args.checksum + workers, 0ULL);
    if (total == 0) std::puts("unexpected zero checksum");
}

} // namespace

int main()
{
    const int workers = std::min(38, kupl_get_num_executors());
    if (workers <= 0 || workers > kMaxWorkers) {
        std::printf("FAIL: invalid KUPL executor count %d\n", workers);
        return 1;
    }
    std::vector<int> executors(workers);
    std::iota(executors.begin(), executors.end(), 0);
    kupl_egroup_h group = kupl_egroup_create(executors.data(), workers);
    if (!group) {
        std::puts("FAIL: kupl_egroup_create");
        return 1;
    }

    std::printf("KUPL executors used: %d\n", workers);
    std::puts("work=0 isolates team wakeup and loop scheduling overhead");
    run_case(512, 0, workers, group);
    run_case(8192, 0, workers, group);
    std::puts("work=256 approximates a non-empty fine-grained page task");
    run_case(512, 256, workers, group);
    run_case(8192, 256, workers, group);

    kupl_egroup_destroy(group);
    return 0;
}
