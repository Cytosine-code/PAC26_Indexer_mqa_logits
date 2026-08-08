// probe_sme_issue_mix.cpp
//
// 目的：判断 2x2 BFMOPA 生产内核只到 ~50-64% 纯 BFMOPA 峰值，是否由
// "循环体内非 BFMOPA 指令挤占发射带宽" 造成，而不是 Load 延迟。
//
// 背景：报告 3.9 (V8) 已经测过"重排 Load/BFMOPA 顺序隐藏延迟"，只有
// +0.86% / +0.15%，说明延迟已被乱序执行掩盖。本探针换一个假设：
// 瓶颈是指令**条数**。probe_sme_peak 的 k 循环只有 6 条指令
// (4 bfmopa + subs + b.ne)，而 pac_sme_page_scores_2x2 是 15 条。
//
// 五个变体，逐级把真实内核的指令加进同一个 4xBFMOPA 循环体：
//   A: 4 bfmopa + subs + b.ne                     (6 条 = probe_sme_peak 结构)
//   B: A + 4 条 ld1h（地址固定，不递增）           (10 条，加 Load 无地址运算)
//   C: B + 3 条地址 add（步长同生产内核）          (13 条)
//   D: C + prfm + add x19                         (15 条 = 生产内核真实组合)
//   E: D 展开 x2，Q 用 #mul vl 立即数偏移，
//      prfm 用 [x19,#64] 立即数偏移               (24 条 / 8 bfmopa)
//
// C/D/E 采用两级循环：内层恰好 64 次、地址步长与 pac_sme_page_scores_2x2
// 逐条一致（x7 += 256, x8 += 64, x9 += 64），外层重置基址。这样内层循环体
// 与生产内核逐指令等价，不需要靠地址回绕掩码来保证 L1 命中。
// E 的内层为 32 次、步长加倍，ZA 累加顺序仍是 pair 0,1,...,63。
//
// 每线程工作集 16 KiB operands + 4 KiB drain = 20 KiB，位于 32 KiB L1D 内，
// 与生产内核的 packed_q/packed_k/page_scores 量级一致。
//
// 读法：
//   B/A ~ 100%  -> Load 不是瓶颈（预期，与 V8 结论一致）
//   D/A ~  60%  -> 发射带宽受限假设成立
//   E/D         -> 展开 x2 在生产内核里能拿到的收益上限
//
// 编译（远程 ARM）：
//   g++ -O3 -march=armv9-a+sme+sve2 -fopenmp probe_sme_issue_mix.cpp \
//       -o probe_sme_issue_mix
//   OMP_NUM_THREADS=38 OMP_PROC_BIND=close taskset -c 0-37 ./probe_sme_issue_mix

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <omp.h>
#include <sched.h>

#include <arm_sve.h>

// operands: >= 16384 bytes, L1 驻留, 线程私有。out: >= 4096 bytes。
// x1 = 外层循环次数（A/B 为唯一循环）。
extern "C" void sme_mix_a(const uint16_t *operands, int64_t iters, float *out);
extern "C" void sme_mix_b(const uint16_t *operands, int64_t iters, float *out);
extern "C" void sme_mix_c(const uint16_t *operands, int64_t iters, float *out);
extern "C" void sme_mix_d(const uint16_t *operands, int64_t iters, float *out);
extern "C" void sme_mix_e(const uint16_t *operands, int64_t iters, float *out);

asm(R"(
    .arch armv9-a+sme+sve2
    .text

// ---- 公共尾部：把 4 个 ZA tile 各自 16 行写出，防止死代码消除 --------------
// 约定：进入时 x2 = out 指针（>= 4096 字节）。使用 x3..x6, w12, p2。
.macro PAC_DRAIN_ZA
    ptrue p2.s
    mov x3, x2
    mov w12, #0
90:
    add x4, x3, #1024
    add x5, x3, #2048
    add x6, x3, #3072
    st1w {za0h.s[w12, 0]}, p2, [x3]
    st1w {za1h.s[w12, 0]}, p2, [x4]
    st1w {za2h.s[w12, 0]}, p2, [x5]
    st1w {za3h.s[w12, 0]}, p2, [x6]
    add x3, x3, #64
    add w12, w12, #1
    cmp w12, #16
    b.lo 90b
.endm

.macro PAC_PROLOGUE
    sub sp, sp, #80
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    str x19, [sp, #64]
.endm

.macro PAC_EPILOGUE
    ldp d8, d9, [sp, #0]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldr x19, [sp, #64]
    add sp, sp, #80
    ret
.endm

// ===========================================================================
// A: 纯 BFMOPA 峰值。Load 提到循环外。
//    循环体 = 4 bfmopa + subs + b.ne = 6 条，每 BFMOPA 0.5 条非 BFMOPA
// ===========================================================================
    .align 4
    .global sme_mix_a
    .type sme_mix_a, %function
sme_mix_a:
    PAC_PROLOGUE
    smstart
    ptrue p0.h
    ptrue p1.h
    zero {za}
    ld1h {z0.h}, p1/z, [x0]
    ld1h {z1.h}, p1/z, [x0, #1, mul vl]
    ld1h {z4.h}, p1/z, [x0, #2, mul vl]
    ld1h {z5.h}, p1/z, [x0, #3, mul vl]
1:
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    subs x1, x1, #1
    b.ne 1b
    PAC_DRAIN_ZA
    smstop
    PAC_EPILOGUE
    .size sme_mix_a, .-sme_mix_a

// ===========================================================================
// B: A + 4 条 ld1h，地址固定不递增（恒 L1 命中，无地址运算）。
//    循环体 = 4 ld1h + 4 bfmopa + subs + b.ne = 10 条
//    三个流地址互不相同，避免硬件把重复地址的 Load 合并/转发。
// ===========================================================================
    .align 4
    .global sme_mix_b
    .type sme_mix_b, %function
sme_mix_b:
    PAC_PROLOGUE
    smstart
    ptrue p0.h
    ptrue p1.h
    zero {za}
    mov x7, x0              // Q
    add x8, x0, #8192       // K tile A
    add x9, x0, #12288      // K tile B
2:
    ld1h {z0.h}, p1/z, [x7]
    ld1h {z1.h}, p1/z, [x7, #1, mul vl]
    ld1h {z4.h}, p1/z, [x8]
    ld1h {z5.h}, p1/z, [x9]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    subs x1, x1, #1
    b.ne 2b
    PAC_DRAIN_ZA
    smstop
    PAC_EPILOGUE
    .size sme_mix_b, .-sme_mix_b

// ===========================================================================
// C: B + 3 条地址 add。内层 64 次，步长与 pac_sme_page_scores_2x2 完全一致：
//    Q += 256, K_A += 64, K_B += 64。外层重置基址。
//    内层循环体 = 4 ld1h + 4 bfmopa + 3 add + subs + b.ne = 13 条
//    地址范围：Q 0..16255, K_A 8192..12223, K_B 12288..16319（均在 16 KiB 内）
// ===========================================================================
    .align 4
    .global sme_mix_c
    .type sme_mix_c, %function
sme_mix_c:
    PAC_PROLOGUE
    smstart
    ptrue p0.h
    ptrue p1.h
    zero {za}
30:
    mov x7, x0
    add x8, x0, #8192
    add x9, x0, #12288
    mov x11, #64
31:
    ld1h {z0.h}, p1/z, [x7]
    ld1h {z1.h}, p1/z, [x7, #1, mul vl]
    ld1h {z4.h}, p1/z, [x8]
    ld1h {z5.h}, p1/z, [x9]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    add x7, x7, #256
    add x8, x8, #64
    add x9, x9, #64
    subs x11, x11, #1
    b.ne 31b
    subs x1, x1, #1
    b.ne 30b
    PAC_DRAIN_ZA
    smstop
    PAC_EPILOGUE
    .size sme_mix_c, .-sme_mix_c

// ===========================================================================
// D: C + prfm + add x19 = pac_sme_page_scores_2x2 的 k 循环真实组合。
//    内层循环体 = 4 ld1h + 4 bfmopa + 3 add + prfm + add + subs + b.ne = 15 条
//    prfm 游标 0..4032，落在同一 L1 驻留缓冲内，不掺入 DRAM 延迟
//    （只测发射代价；生产内核的 prfm 打的是 DRAM，那是另一回事）。
// ===========================================================================
    .align 4
    .global sme_mix_d
    .type sme_mix_d, %function
sme_mix_d:
    PAC_PROLOGUE
    smstart
    ptrue p0.h
    ptrue p1.h
    zero {za}
40:
    mov x7, x0
    add x8, x0, #8192
    add x9, x0, #12288
    mov x19, x0
    mov x11, #64
41:
    ld1h {z0.h}, p1/z, [x7]
    ld1h {z1.h}, p1/z, [x7, #1, mul vl]
    ld1h {z4.h}, p1/z, [x8]
    ld1h {z5.h}, p1/z, [x9]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    add x7, x7, #256
    add x8, x8, #64
    add x9, x9, #64
    prfm pldl2keep, [x19]
    add x19, x19, #64
    subs x11, x11, #1
    b.ne 41b
    subs x1, x1, #1
    b.ne 40b
    PAC_DRAIN_ZA
    smstop
    PAC_EPILOGUE
    .size sme_mix_d, .-sme_mix_d

// ===========================================================================
// E: D 展开 x2。内层 32 次，每次 8 条 BFMOPA。
//    Q 的第二组用 #4/#5 mul vl 立即数偏移（VL=64B，#4 mul vl = +256），
//    K 的第二组用 #1 mul vl，prfm 第二条用 [x19, #64]，省掉 4 条 add。
//    内层循环体 = 8 ld1h + 8 bfmopa + 3 add + 2 prfm + add + subs + b.ne = 24 条
//    每 BFMOPA 非 BFMOPA 指令数：D 2.75 -> E 2.00
//    ZA 累加顺序仍是 pair 0,1,...,63，与生产内核逐位一致。
// ===========================================================================
    .align 4
    .global sme_mix_e
    .type sme_mix_e, %function
sme_mix_e:
    PAC_PROLOGUE
    smstart
    ptrue p0.h
    ptrue p1.h
    zero {za}
50:
    mov x7, x0
    add x8, x0, #8192
    add x9, x0, #12288
    mov x19, x0
    mov x11, #32
51:
    ld1h {z0.h}, p1/z, [x7]
    ld1h {z1.h}, p1/z, [x7, #1, mul vl]
    ld1h {z4.h}, p1/z, [x8]
    ld1h {z5.h}, p1/z, [x9]
    ld1h {z2.h}, p1/z, [x7, #4, mul vl]
    ld1h {z3.h}, p1/z, [x7, #5, mul vl]
    ld1h {z6.h}, p1/z, [x8, #1, mul vl]
    ld1h {z7.h}, p1/z, [x9, #1, mul vl]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    bfmopa za0.s, p0/m, p0/m, z2.h, z6.h
    bfmopa za1.s, p0/m, p0/m, z2.h, z7.h
    bfmopa za2.s, p0/m, p0/m, z3.h, z6.h
    bfmopa za3.s, p0/m, p0/m, z3.h, z7.h
    add x7, x7, #512
    add x8, x8, #128
    add x9, x9, #128
    prfm pldl2keep, [x19]
    prfm pldl2keep, [x19, #64]
    add x19, x19, #128
    subs x11, x11, #1
    b.ne 51b
    subs x1, x1, #1
    b.ne 50b
    PAC_DRAIN_ZA
    smstop
    PAC_EPILOGUE
    .size sme_mix_e, .-sme_mix_e
)");

namespace {

constexpr int64_t OPERAND_BYTES = 16384;   // 16 KiB，同 packed_q / packed_k
constexpr int64_t OPERAND_HALVES = OPERAND_BYTES / 2;
constexpr int64_t OUT_FLOATS = 1024;       // 4 KiB drain 目标

struct Variant {
    const char *name;
    void (*fn)(const uint16_t *, int64_t, float *);
    // 每个 x1 值对应的 BFMOPA 条数：A/B 为 4；C/D 内层 64x4；E 内层 32x8。
    int64_t bfmopa_per_outer;
    const char *body;
    double non_bfmopa_per_bfmopa;
};

double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main()
{
    const int svl_bytes = static_cast<int>(svcnth()) * 2;
    if (svl_bytes != 64) {
        std::printf("FAIL: expected 512-bit SME SVL, got %d bits\n",
                    svl_bytes * 8);
        return 1;
    }

    const Variant variants[] = {
        {"A pure-bfmopa   ", sme_mix_a,   4, "4 bfmopa + subs/b.ne          (6)", 0.50},
        {"B +4 ld1h       ", sme_mix_b,   4, "+ loads, no addr math        (10)", 1.50},
        {"C +3 addr add   ", sme_mix_c, 256, "+ addr increments            (13)", 2.25},
        {"D +prfm  [REAL] ", sme_mix_d, 256, "= production 2x2 k-loop      (15)", 2.75},
        {"E D unrolled x2 ", sme_mix_e, 256, "unroll x2, imm offsets  (24 / 8)", 2.00},
    };

    const int threads = omp_get_max_threads();

    // 绑核自检：报告 4.2 记过 taskset 错一个 CPU 编号就掉数 TFLOPS。
    std::vector<int> cpus(threads, -1);
#pragma omp parallel
    {
        cpus[omp_get_thread_num()] = sched_getcpu();
    }
    std::vector<int> sorted_cpus = cpus;
    std::sort(sorted_cpus.begin(), sorted_cpus.end());
    const int unique_cpus = static_cast<int>(
        std::unique(sorted_cpus.begin(), sorted_cpus.end()) -
        sorted_cpus.begin());

    // 缓冲在计时区外分配，并由各自线程 first-touch，保证 NUMA 本地。
    std::vector<uint16_t> operand_pool(
        static_cast<size_t>(threads) * OPERAND_HALVES);
    std::vector<float> out_pool(
        static_cast<size_t>(threads) * OUT_FLOATS);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        uint16_t *ops = operand_pool.data() + tid * OPERAND_HALVES;
        for (int64_t i = 0; i < OPERAND_HALVES; ++i) {
            // bf16 值约 1.0~2.0，避免累加溢出成 inf 影响后续读数。
            ops[i] = static_cast<uint16_t>(0x3f80 | (i & 0x3f));
        }
        float *out = out_pool.data() + tid * OUT_FLOATS;
        for (int64_t i = 0; i < OUT_FLOATS; ++i) {
            out[i] = 0.0f;
        }
    }

    // 每变体 BFMOPA 总数一致（~20M/线程 => ~20 ms），便于直接比 TFLOPS，
    // 也让 OpenMP region 创建（~9.4 us）摊薄到 0.05% 以下。
    const int64_t bfmopa_total = 20000000;
    const int runs = 7;

    std::printf("SME SVL: %d bits\n", svl_bytes * 8);
    std::printf("threads=%d unique_cpus=%d cpu_range=%d-%d\n",
                threads, unique_cpus, sorted_cpus.front(),
                sorted_cpus[unique_cpus - 1]);
    std::printf("bfmopa/thread=%lld  runs=%d (median)\n\n",
                static_cast<long long>(bfmopa_total), runs);

    std::printf("%-17s %-34s %6s %9s %7s\n",
                "variant", "loop body (instr)", "n/B", "TFLOPS", "vs A");
    std::printf("---------------------------------------------------------"
                "----------------------\n");

    double baseline = 0;
    for (const Variant &v : variants) {
        const int64_t outer = bfmopa_total / v.bfmopa_per_outer;
        const double actual_bfmopa =
            static_cast<double>(outer) * v.bfmopa_per_outer;
        // 每条 BFMOPA 1024 FLOP（16x16 FP32 tile, K=2）。
        const double flops = actual_bfmopa * 1024.0 * threads;

        std::vector<double> tflops;
        for (int run = -1; run < runs; ++run) {   // run -1 = warmup
            const double begin = omp_get_wtime();
#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                v.fn(operand_pool.data() + tid * OPERAND_HALVES, outer,
                     out_pool.data() + tid * OUT_FLOATS);
            }
            const double elapsed = omp_get_wtime() - begin;
            if (run >= 0 && elapsed > 0) {
                tflops.push_back(flops / elapsed / 1e12);
            }
        }

        const double med = median_of(tflops);
        if (baseline == 0) {
            baseline = med;
        }
        std::printf("%-17s %-34s %6.2f %9.3f %6.1f%%\n",
                    v.name, v.body, v.non_bfmopa_per_bfmopa, med,
                    med / baseline * 100.0);
    }

    // 阻止 out_pool 被优化掉。
    double sink = 0;
    for (int t = 0; t < threads; ++t) {
        sink += out_pool[static_cast<size_t>(t) * OUT_FLOATS];
    }

    std::printf("\nn/B = 每条 BFMOPA 摊到的非 BFMOPA 指令数\n\n");
    std::printf("读法:\n");
    std::printf("  B/A ~ 100%%  -> Load 不是瓶颈（应与 V8 的 +0.86%% 结论一致）\n");
    std::printf("  D/A ~  60%%  -> 发射带宽受限成立，非 BFMOPA 指令在抢发射槽\n");
    std::printf("  E/D         -> 展开 x2 在 pac_sme_page_scores_2x2 的收益上限\n");
    std::printf("  若 D/A ~ 100%%，则瓶颈不在内核发射，应回到内存侧找\n");
    std::printf("\n(checksum %.3g)\n", sink);
    return 0;
}
