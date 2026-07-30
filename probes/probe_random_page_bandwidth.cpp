#include <arm_sve.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <omp.h>
#include <random>
#include <sched.h>
#include <sys/mman.h>
#include <vector>

#include <numa.h>
#include <numaif.h>

constexpr size_t kPageBytes = 16 * 1024;
constexpr size_t kCacheLine = 64;

static size_t env_size(const char *name, size_t fallback)
{
    const char *text = std::getenv(name);
    if (!text || !*text) {
        return fallback;
    }
    return static_cast<size_t>(std::strtoull(text, nullptr, 10));
}

__attribute__((noinline))
static uint64_t read_page_linear(const uint8_t *page)
{
    const svbool_t pg = svptrue_b64();
    svuint64_t acc0 = svdup_u64(0);
    svuint64_t acc1 = svdup_u64(0);
    svuint64_t acc2 = svdup_u64(0);
    svuint64_t acc3 = svdup_u64(0);
    for (size_t offset = 0; offset < kPageBytes; offset += 4 * kCacheLine) {
        acc0 = sveor_u64_x(pg, acc0, svld1_u64(
            pg, reinterpret_cast<const uint64_t *>(page + offset)));
        acc1 = sveor_u64_x(pg, acc1, svld1_u64(
            pg, reinterpret_cast<const uint64_t *>(page + offset + 64)));
        acc2 = sveor_u64_x(pg, acc2, svld1_u64(
            pg, reinterpret_cast<const uint64_t *>(page + offset + 128)));
        acc3 = sveor_u64_x(pg, acc3, svld1_u64(
            pg, reinterpret_cast<const uint64_t *>(page + offset + 192)));
    }
    const svuint64_t merged = sveor_u64_x(
        pg, sveor_u64_x(pg, acc0, acc1), sveor_u64_x(pg, acc2, acc3));
    return svaddv_u64(pg, merged);
}

__attribute__((noinline))
static uint64_t read_page_k_pack_order(const uint8_t *page)
{
    const svbool_t pg = svptrue_b64();
    svuint64_t acc0 = svdup_u64(0);
    svuint64_t acc1 = svdup_u64(0);
    svuint64_t acc2 = svdup_u64(0);
    svuint64_t acc3 = svdup_u64(0);
    for (size_t token_block = 0; token_block < 4; ++token_block) {
        for (size_t column = 0; column < 4; ++column) {
            const size_t base = token_block * 4096 + column * 64;
            for (size_t row = 0; row < 16; row += 4) {
                acc0 = sveor_u64_x(pg, acc0, svld1_u64(
                    pg, reinterpret_cast<const uint64_t *>(
                        page + base + (row + 0) * 256)));
                acc1 = sveor_u64_x(pg, acc1, svld1_u64(
                    pg, reinterpret_cast<const uint64_t *>(
                        page + base + (row + 1) * 256)));
                acc2 = sveor_u64_x(pg, acc2, svld1_u64(
                    pg, reinterpret_cast<const uint64_t *>(
                        page + base + (row + 2) * 256)));
                acc3 = sveor_u64_x(pg, acc3, svld1_u64(
                    pg, reinterpret_cast<const uint64_t *>(
                        page + base + (row + 3) * 256)));
            }
        }
    }
    const svuint64_t merged = sveor_u64_x(
        pg, sveor_u64_x(pg, acc0, acc1), sveor_u64_x(pg, acc2, acc3));
    return svaddv_u64(pg, merged);
}

using PageReader = uint64_t (*)(const uint8_t *);

struct Result {
    double seconds;
    uint64_t checksum;
};

static Result benchmark_pages(
    const uint8_t *data, const std::vector<uint32_t> &order,
    size_t passes, bool random_pages, PageReader reader)
{
    const size_t pages = order.size();
    const int max_threads = omp_get_max_threads();
    std::vector<uint64_t> checksums(max_threads, 0);
    const double start = omp_get_wtime();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        const size_t begin = pages * static_cast<size_t>(tid) / threads;
        const size_t end = pages * static_cast<size_t>(tid + 1) / threads;
        uint64_t checksum = 0;
        for (size_t pass = 0; pass < passes; ++pass) {
            for (size_t i = begin; i < end; ++i) {
                const size_t physical = random_pages ? order[i] : i;
                checksum += reader(data + physical * kPageBytes);
            }
        }
        checksums[tid] = checksum;
    }
    const double stop = omp_get_wtime();
    return {stop - start,
            std::accumulate(checksums.begin(), checksums.end(), uint64_t(0))};
}

static void report_series(
    const char *name, const uint8_t *data,
    const std::vector<uint32_t> &order, size_t passes,
    int runs, bool random_pages, PageReader reader)
{
    const double bytes = static_cast<double>(order.size()) *
        static_cast<double>(kPageBytes) * static_cast<double>(passes);
    std::vector<double> bandwidths;
    bandwidths.reserve(runs);
    uint64_t checksum = 0;
    std::printf("%s\n", name);
    for (int run = 0; run < runs; ++run) {
        const Result result = benchmark_pages(
            data, order, passes, random_pages, reader);
        const double gbps = bytes / result.seconds / 1.0e9;
        bandwidths.push_back(gbps);
        checksum ^= result.checksum;
        std::printf("  run %d: %.3f ms  %.2f GB/s\n",
                    run + 1, result.seconds * 1.0e3, gbps);
    }
    std::sort(bandwidths.begin(), bandwidths.end());
    const double median = bandwidths[bandwidths.size() / 2];
    std::printf("  median: %.2f GB/s\n", median);
    std::printf("  best:   %.2f GB/s\n", bandwidths.back());
    std::printf("  checksum: 0x%016llx\n",
                static_cast<unsigned long long>(checksum));
    if (random_pages && reader == read_page_k_pack_order) {
        std::printf("  Case1 compulsory-K roof: %.3f TFLOPS\n",
                    median * 128.0 / 1000.0);
        std::printf("  Case2 compulsory-K roof: %.3f TFLOPS\n",
                    median * 64.0 / 1000.0);
    }
}

int main()
{
    size_t bytes = env_size("PAGE_PROBE_BYTES", 512ULL * 1024 * 1024);
    bytes = bytes / kPageBytes * kPageBytes;
    const size_t passes = env_size("PAGE_PROBE_PASSES", 8);
    const int runs = static_cast<int>(env_size("PROBE_RUNS", 7));
    if (bytes < kPageBytes || passes == 0 || runs <= 0) {
        std::puts("FAIL: invalid probe size, passes, or runs");
        return 2;
    }

    const int main_cpu = sched_getcpu();
    const int cpu_node = numa_node_of_cpu(main_cpu);
    const int memory_node = cpu_node + 16;
    void *mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (mapping == MAP_FAILED) {
        std::perror("mmap");
        return 2;
    }
    unsigned long mask = 1UL << memory_node;
    if (mbind(mapping, bytes, MPOL_BIND, &mask, sizeof(mask) * 8,
              MPOL_MF_STRICT | MPOL_MF_MOVE) != 0) {
        std::fprintf(stderr, "mbind node %d failed: %s\n",
                     memory_node, std::strerror(errno));
        munmap(mapping, bytes);
        return 2;
    }
    auto *data = static_cast<uint8_t *>(mapping);

    const int max_threads = omp_get_max_threads();
    std::vector<int> cpus(max_threads, -1);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        cpus[tid] = sched_getcpu();
#pragma omp for schedule(static)
        for (size_t offset = 0; offset < bytes; offset += kCacheLine) {
            *reinterpret_cast<uint64_t *>(data + offset) =
                static_cast<uint64_t>(offset / kCacheLine + 1);
        }
    }
    std::sort(cpus.begin(), cpus.end());
    const int unique_cpus = static_cast<int>(
        std::unique(cpus.begin(), cpus.end()) - cpus.begin());

    const size_t pages = bytes / kPageBytes;
    std::vector<uint32_t> order(pages);
    std::iota(order.begin(), order.end(), uint32_t(0));
    std::mt19937 rng(20260730);
    std::shuffle(order.begin(), order.end(), rng);

    std::printf("threads=%d unique_cpus=%d cpu_range=%d-%d\n",
                max_threads, unique_cpus, cpus.front(), cpus.back());
    std::printf("main_cpu=%d cpu_node=%d bound_memory_node=%d\n",
                main_cpu, cpu_node, memory_node);
    std::printf("dataset=%.1f MiB pages=%zu passes=%zu payload=%.1f GiB/run\n",
                bytes / 1048576.0, pages, passes,
                static_cast<double>(bytes) * passes / 1073741824.0);

    benchmark_pages(data, order, 1, true, read_page_k_pack_order);
    report_series("sequential pages + linear cache lines:",
                  data, order, passes, runs, false, read_page_linear);
    report_series("random pages + K-pack access order:",
                  data, order, passes, runs, true, read_page_k_pack_order);

    munmap(mapping, bytes);
    return 0;
}
