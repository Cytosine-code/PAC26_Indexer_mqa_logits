// probe_page_bw_matrix.cpp
//
// 目的：研究 Case 2 是否已到带宽极限。
//
// 第一版探针因两个结构缺陷测出了失真的数（S-LIN=160 而非 385）：
//   1. 读循环用单个 XOR 累加器，串行依赖链压缩乱序窗口，在途 load 太少；
//      旧探针用 4 个独立累加器，MLP 高约 2.4x。修复：读函数照抄旧探针。
//   2. 数据用 std::vector 分配，未 mbind 到 node+16；生产 kv_cache 走
//      mmap_on_package_memory（绑定 node+16）。修复：mmap + mbind 同旧探针。
//
// 修复后校准位已恢复（上轮实测）：
//   S-LIN 390 GB/s（旧 385）   S-KPK 348   R-KPK 204（旧 206）
//
// 上轮结论（probe_page_bw_matrix.md）：
//   - R-LIN=347 GB/s：随机页映射几乎免费（S-LIN 390 vs R-LIN 347，仅 -11%）。
//     生产 V12 的 PRFM 流正是"随机页 + 页内线性"，所以 347 是生产 DRAM
//     模式的硬件上限 = Case 2 的 22.2 TFLOPS。
//   - PACE（随机+跨行+预取 1 页）= 225 ≈ 生产 219：当前预取实现就停在这，
//     是 PRFM 这条路的天花板，不是 DRAM 的墙。
//
// 本版新增三个判据变体，拆开"225 与 347 之间的缺口"：
//   PACE-LIN（线性读 + 预取）: 读若命中 L2，DRAM 流量只剩 PRFM 流
//        -> ~347 表示 PRFM 没问题、跨行读没全命中 L2
//        -> ~225 表示 PRFM 流天生只有 ~225
//   PACE x2 / x4（预取深度 2/4）: 测预取流水深度是否为杠杆
//        -> 爬向 347 表示深度是杠杆，生产在 SME 窗口加深即可
//        -> 停 225 表示深度无用，PRFM 就是 225 的墙
//
// 口径提醒：探针不含 BFMOPA。生产 219 GB/s 是 SME 掩护下达成，PACE 模式
// 用"预取下页 + L2 命中读当前页"复刻同一节奏，两者才是可比的。
//
// 编译（远程 ARM）：
//   g++ -O3 -march=armv9-a+sme+sve2 -fopenmp probe_page_bw_matrix.cpp \
//       -o probe_page_bw_matrix
//   OMP_NUM_THREADS=38 OMP_PROC_BIND=close taskset -c 0-37 ./probe_page_bw_matrix
//   （探针通过 numa_node_of_cpu()+16 绑定内存，与生产 mmap_on_package_memory 一致）

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

namespace {

constexpr size_t kPageBytes = 16 * 1024;
constexpr size_t kCacheLine = 64;

// ---------------------------------------------------------------------------
// 读函数：逐字照抄旧探针（probe_random_page_bandwidth.cpp），4 个独立累加器
// 保证 MLP。这是 385/206 数字成立的结构前提。
// ---------------------------------------------------------------------------
__attribute__((noinline))
uint64_t read_page_linear(const uint8_t *page)
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
uint64_t read_page_k_pack_order(const uint8_t *page)
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

// 与生产 V12 相同的下页预取：64B 线性推进，pldl2keep。
__attribute__((noinline))
void prefetch_page_linear(const uint8_t *page)
{
    for (size_t offset = 0; offset < kPageBytes; offset += kCacheLine) {
        __asm__ __volatile__("prfm pldl2keep, [%0]"
                             :: "r"(page + offset));
    }
}

using PageReader = uint64_t (*)(const uint8_t *);

enum class Order { LINEAR, KPACK };

struct Mode {
    const char *name;
    bool random_pages;
    Order order;
    int prefetch_depth;     // 0=不预取; k=读当前页前预取 i+k 页（生产复刻/深度实验）
};

struct Result {
    double seconds;
    uint64_t checksum;
};

static Result benchmark_pages(
    const uint8_t *data, const std::vector<uint32_t> &order,
    size_t passes, const Mode &mode)
{
    const size_t pages = order.size();
    const int max_threads = omp_get_max_threads();
    std::vector<uint64_t> checksums(max_threads, 0);
    const PageReader reader =
        mode.order == Order::KPACK ? read_page_k_pack_order
                                   : read_page_linear;
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
                const size_t physical =
                    mode.random_pages ? order[i] : i;
                const uint8_t *page = data + physical * kPageBytes;
                if (mode.prefetch_depth > 0) {
                    const size_t pf = i + static_cast<size_t>(mode.prefetch_depth);
                    if (pf < end) {
                        const size_t pf_phys =
                            mode.random_pages ? order[pf] : pf;
                        prefetch_page_linear(data + pf_phys * kPageBytes);
                    }
                }
                checksum += reader(page);
            }
        }
        checksums[tid] = checksum;
    }
    const double stop = omp_get_wtime();
    return {stop - start,
            std::accumulate(checksums.begin(), checksums.end(), uint64_t(0))};
}

void report_series(
    const char *name, const uint8_t *data,
    const std::vector<uint32_t> &order, size_t passes,
    int runs, const Mode &mode)
{
    const double bytes = static_cast<double>(order.size()) *
        static_cast<double>(kPageBytes) * static_cast<double>(passes);
    std::vector<double> bandwidths;
    bandwidths.reserve(static_cast<size_t>(runs));
    uint64_t checksum = 0;
    std::printf("%s\n", name);
    for (int run = 0; run < runs; ++run) {
        const Result result = benchmark_pages(data, order, passes, mode);
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
    std::printf("\n");
}

}  // namespace

int main()
{
    size_t bytes = 512ULL * 1024 * 1024;
    bytes = bytes / kPageBytes * kPageBytes;
    const size_t passes = 8;
    const int runs = 7;

    // 内存绑定：与生产 mmap_on_package_memory 完全一致
    // （numa_node_of_cpu + 16，MPOL_BIND + MAP_POPULATE）。
    const int main_cpu = sched_getcpu();
    const int cpu_node = numa_node_of_cpu(main_cpu);
    const int memory_node = cpu_node + 16;
    void *mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (mapping == MAP_FAILED) {
        std::perror("mmap");
        return 2;
    }
    unsigned long mask = 1UL << static_cast<unsigned long>(memory_node);
    if (mbind(mapping, bytes, MPOL_BIND, &mask, sizeof(mask) * 8,
              MPOL_MF_STRICT | MPOL_MF_MOVE) != 0) {
        std::fprintf(stderr, "mbind node %d failed: %s\n",
                     memory_node, std::strerror(errno));
        munmap(mapping, bytes);
        return 2;
    }
    auto *data = static_cast<uint8_t *>(mapping);

    const int max_threads = omp_get_max_threads();
    std::vector<int> cpus(static_cast<size_t>(max_threads), -1);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        cpus[static_cast<size_t>(tid)] = sched_getcpu();
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
    std::printf("\n");

    const Mode modes[] = {
        // 校准位
        {"S-LIN   seq + linear             ", false, Order::LINEAR, 0},
        {"S-KPK   seq + K-pack             ", false, Order::KPACK,  0},
        {"R-LIN   rand + linear            ", true,  Order::LINEAR, 0},
        {"R-KPK   rand + K-pack            ", true,  Order::KPACK,  0},
        // 生产复刻（深度 1，上轮实测 225 ≈ 生产 219）
        {"PACE    rand + K-pack + PRFM x1  ", true,  Order::KPACK,  1},
        // 判据：跨行读是否被 L2 隐藏（读若命中 L2，DRAM 流量只剩 PRFM）
        {"PACE-LIN rand + linear + PRFM x1 ", true,  Order::LINEAR, 1},
        // 判据：预取流水深度是否为杠杆
        {"PACE    rand + K-pack + PRFM x2  ", true,  Order::KPACK,  2},
        {"PACE    rand + K-pack + PRFM x4  ", true,  Order::KPACK,  4},
    };
    const size_t num_modes = sizeof(modes) / sizeof(modes[0]);

    for (size_t m = 0; m < num_modes; ++m) {
        report_series(modes[m].name, data, order, passes, runs, modes[m]);
    }

    // Case 2 的 K 读取算术强度固定 64 FLOP/Byte（next_n=1）。
    std::printf("========================================================\n");
    std::printf("Case 2 roof (K-read arithmetic intensity = 64 FLOP/Byte):\n");
    std::printf("  R-LIN（生产 DRAM 模式上限）347 GB/s => 22.2 TFLOPS\n");
    std::printf("  PACE x1（生产复刻）         225 GB/s => 14.4 TFLOPS\n");
    std::printf("  V12 production              ~219 GB/s => 14.0 TFLOPS\n");
    std::printf("\n判读（225 是预取实现的天花板，347 是硬件能力）:\n");
    std::printf("  PACE-LIN ~ 347  -> 线性读喂饱 L2，PRFM 没问题，跨行读未全命中\n");
    std::printf("  PACE-LIN ~ 225  -> PRFM 流天生只有 ~225，换消费路径才够到 347\n");
    std::printf("  PACE x2/x4 爬向 347 -> 预取流水太浅，生产在 SME 窗口加深即可\n");
    std::printf("  PACE x2/x4 停 225   -> 深度无用，PRFM 就是 225 的墙\n");
    std::printf("========================================================\n");

    munmap(mapping, bytes);
    return 0;
}
