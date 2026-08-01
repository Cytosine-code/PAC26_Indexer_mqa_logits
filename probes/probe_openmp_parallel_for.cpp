#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <sched.h>
#include <vector>

#include <omp.h>

namespace {
constexpr int kWorkers = 38;
constexpr int kSamples = 101;
alignas(64) unsigned long long checksum[256] = {};
alignas(64) int cpus[256] = {};

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

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void run_case(size_t pages, int work)
{
    std::vector<double> times;
    for (int sample = -10; sample < kSamples; ++sample) {
        const double begin = now_us();
#pragma omp parallel num_threads(kWorkers)
        {
            const int tid = omp_get_thread_num();
            unsigned long long sum = checksum[tid];
#pragma omp for schedule(static)
            for (size_t page = 0; page < pages; ++page) {
                sum += do_page_work(page, work);
            }
            checksum[tid] = sum;
            cpus[tid] = sched_getcpu();
        }
        const double elapsed = now_us() - begin;
        if (sample >= 0) times.push_back(elapsed);
    }
    std::printf("OpenMP pages=%zu work=%d: %.2f us\n",
        pages, work, median(times));
    std::printf("  CPUs:");
    for (int i = 0; i < kWorkers; ++i) std::printf(" %d", cpus[i]);
    std::putchar('\n');
    if (std::accumulate(checksum, checksum + kWorkers, 0ULL) == 0)
        std::puts("unexpected zero checksum");
}
} // namespace

int main()
{
    std::printf("OpenMP workers used: %d\n", kWorkers);
    run_case(512, 0);
    run_case(8192, 0);
    run_case(512, 256);
    run_case(8192, 256);
    return 0;
}
