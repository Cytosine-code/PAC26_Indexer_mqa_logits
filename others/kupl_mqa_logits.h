#pragma once

#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <omp.h>

#include "indexer_mqa_logits.h"

#ifdef __aarch64__
#include "kupl.h"
#endif

// kupl_bf16_paged_mqa_logits：与 indexer_bf16_paged_mqa_logits 计算路径逐字节相同的
// KUPL 调度版姊妹内核。唯一区别在调度层：
//
//   原版： 一个 OpenMP parallel region 内含两阶段
//          （Q Pack omp-for + 隐式 barrier -> 线程私有 Page 区间循环）
//   本版： 一个 kupl_parallel_for(STATIC) 同时完成 Q Pack 和 Page 循环
//
// 为什么合并成一次并行调度：
//   * 原版每个 batch 调用重建 OpenMP region（约 9.4 us 固定开销），中间还有
//     Q Pack 与 Page 阶段的全局 barrier。
//   * KUPL parallel_for 单次启动约 5.5 us。若照搬两阶段，需要两次 parallel_for，
//     启动开销反而翻倍（约 11 us）而更慢。
//   * 本版把 Q Pack 移入 Page 工作线程：每个线程先打包自己 Page 区间所覆盖的
//     batch 的 Q，再处理这些 Page。只有区间边界的 batch 会被相邻两个线程重复打包，
//     写入值完全相同；且每个线程只读取自己打包过的行，不存在跨线程
//     read-before-write 竞态，因此不需要任何全局 barrier。
//
// 计算内核（pac_sme_* 汇编、SME 2×2 BFMOPA、SVE 后处理、Q/K 打包）与
// indexer_mqa_logits.h 完全一致，因此 A/B 只反映调度差异。
//
// 运行环境（与 probes/run_kupl_probes.sh 相同）：
//   export KUPL_EXECUTOR_COUNT=38
//   export KUPL_EXECUTOR_BACKEND=pthread
//   export KUPL_SCHED_POLICY=static_mq
//   g++ -O3 -march=armv9-a+sme+sve2 -fopenmp -lnuma -lkupl main.cpp -o main
//   OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-37 ./main

#ifdef __aarch64__

namespace kupl_mqa_detail {

struct KuplMqaPageCtx {
    const bfloat16_t *q = nullptr;
    const bfloat16_t *kv_cache = nullptr;
    const int64_t *block_tables = nullptr;
    const int64_t *context_lens = nullptr;
    const float *weights = nullptr;
    float *output = nullptr;
    int64_t batch_size = 0;
    int64_t next_n = 0;
    int64_t num_heads = 0;
    int64_t dim = 0;
    int64_t block_size = 0;
    int64_t max_model_len = 0;
    int64_t max_num_blocks = 0;
    int64_t packed_q_stride = 0;
    bfloat16_t *packed_q_all = nullptr;
    int64_t *page_offsets = nullptr;
    int64_t total_pages = 0;
    MqaLogitsPhaseTiming *phase_timing = nullptr;
};

static void kupl_mqa_page_worker(kupl_nd_range_t *nd_range, void *opaque, int tid, int /*tnum*/)
{
    auto *ctx = static_cast<KuplMqaPageCtx *>(opaque);
    const int64_t begin = nd_range->nd_range[0].lower;
    const int64_t end = nd_range->nd_range[0].upper;
    if (begin >= end) {
        return;
    }

    alignas(64) bfloat16_t packed_k[64 * 128];
    alignas(64) float page_scores[64 * 64];

#ifdef EN_TIMING
    double _t_q = 0, _t_k = 0, _t_s = 0, _t_p = 0, _t_m = 0;
    const double _t_thread_start = get_clock_us();
    double _t_phase;
#endif

    // ---- Phase 1: 打包本线程 Page 区间覆盖的所有 Q 行并初始化 mask ----
    // STATIC 策略给每个 worker 一段连续 Page 区间 [begin, end)，其覆盖的
    // batch 范围 [first_batch, last_batch] 只在该线程内使用。边界 batch 由相邻
    // 两个线程重复打包，写入值一致，且各自读自己写的行，安全。
    const int64_t first_batch = static_cast<int64_t>(
        std::upper_bound(ctx->page_offsets, ctx->page_offsets + ctx->batch_size + 1,
                         begin) - ctx->page_offsets - 1);
    const int64_t last_batch = static_cast<int64_t>(
        std::upper_bound(ctx->page_offsets, ctx->page_offsets + ctx->batch_size + 1,
                         end - 1) - ctx->page_offsets - 1);

    const svbool_t pack_pg = svptrue_b32();
    const svuint32_t pack_offsets = svindex_u32(0, 256);
    for (int64_t b = first_batch; b <= last_batch; ++b) {
        const int64_t context_len = ctx->context_lens[b];
        for (int64_t n = 0; n < ctx->next_n; ++n) {
            const int64_t row = b * ctx->next_n + n;
            const bfloat16_t *q_row = ctx->q + row * ctx->packed_q_stride;
            bfloat16_t *packed_q = ctx->packed_q_all + row * ctx->packed_q_stride;

#ifdef EN_TIMING
            _t_phase = get_clock_us();
#endif
            for (int64_t kp = 0; kp < 64; ++kp) {
                for (int64_t hb = 0; hb < 4; ++hb) {
                    bfloat16_t *dst = packed_q + (kp * 4 + hb) * 32;
                    const bfloat16_t *src = q_row + hb * 16 * ctx->dim + kp * 2;
                    const svuint32_t pairs = svld1_gather_u32offset_u32(
                        pack_pg, reinterpret_cast<const uint32_t *>(src), pack_offsets);
                    svst1_u32(pack_pg, reinterpret_cast<uint32_t *>(dst), pairs);
                }
            }
#ifdef EN_TIMING
            _t_q += get_clock_us() - _t_phase;
            _t_phase = get_clock_us();
#endif
            float *out = ctx->output + row * ctx->max_model_len;
            const int64_t valid_length = context_len - ctx->next_n + n + 1;
            std::fill(out + valid_length, out + ctx->max_model_len, -INFINITY);
#ifdef EN_TIMING
            _t_m += get_clock_us() - _t_phase;
#endif
        }
    }

    // ---- Phase 2: 处理 Page 区间 [begin, end)，与 indexer 版逐字节一致 ----
    int64_t batch_idx = first_batch;
    int64_t logical_block = begin - ctx->page_offsets[first_batch];

    for (int64_t page_index = begin; page_index < end; ++page_index) {
        while (batch_idx < ctx->batch_size &&
               logical_block >=
                   ctx->page_offsets[batch_idx + 1] - ctx->page_offsets[batch_idx]) {
            ++batch_idx;
            logical_block = 0;
        }

        const int64_t context_len = ctx->context_lens[batch_idx];
        const int64_t physical_block =
            ctx->block_tables[batch_idx * ctx->max_num_blocks + logical_block];
        const int64_t token_base = logical_block * ctx->block_size;

        if (physical_block < 0) {
            for (int64_t n = 0; n < ctx->next_n; ++n) {
                const int64_t row = batch_idx * ctx->next_n + n;
                const int64_t valid_length = context_len - ctx->next_n + n + 1;
                const int64_t invalid_tokens = std::min<int64_t>(
                    ctx->block_size, valid_length - token_base);
                if (invalid_tokens > 0) {
                    float *out = ctx->output + row * ctx->max_model_len + token_base;
                    std::fill(out, out + invalid_tokens, -INFINITY);
                }
            }
            ++logical_block;
            continue;
        }

        const bfloat16_t *k_page =
            ctx->kv_cache + physical_block * ctx->block_size * ctx->dim;

#ifdef EN_TIMING
        _t_phase = get_clock_us();
        pac_sme_pack_k_page(k_page, packed_k);
        _t_k += get_clock_us() - _t_phase;
#endif

        bool packed_k_ready = false;
        for (int64_t n = 0; n < ctx->next_n; ++n) {
            const int64_t row = batch_idx * ctx->next_n + n;
            const int64_t q_limit = context_len - ctx->next_n + n;
            const int64_t valid_tokens = std::min<int64_t>(
                ctx->block_size, q_limit + 1 - token_base);
            if (valid_tokens <= 0) {
                continue;
            }
            const int64_t token_tiles = ceil_div(valid_tokens, int64_t(16));
            const bfloat16_t *packed_q =
                ctx->packed_q_all + row * ctx->packed_q_stride;

#ifdef EN_TIMING
            _t_phase = get_clock_us();
            if (valid_tokens == 64) {
                pac_sme_page_scores_2x2(packed_q, packed_k, page_scores);
            } else {
                pac_sme_page_scores(packed_q, packed_k, page_scores, token_tiles);
            }
            _t_s += get_clock_us() - _t_phase;
            _t_phase = get_clock_us();
#else
            if (!packed_k_ready) {
                if (valid_tokens == 64) {
                    pac_sme_pack_k_page_and_scores_2x2(
                        k_page, packed_k, packed_q, page_scores);
                } else {
                    pac_sme_pack_k_page_and_scores(
                        k_page, packed_k, packed_q, page_scores, token_tiles);
                }
                packed_k_ready = true;
            } else {
                if (valid_tokens == 64) {
                    pac_sme_page_scores_2x2(
                        packed_q, packed_k, page_scores);
                } else {
                    pac_sme_page_scores(
                        packed_q, packed_k, page_scores, token_tiles);
                }
            }
#endif
            const float *row_weights = ctx->weights + row * ctx->num_heads;
            float *out = ctx->output + row * ctx->max_model_len + token_base;
            const svbool_t all = svptrue_b32();
            const svfloat32_t zero = svdup_f32(0.0f);
            if (token_tiles == 4) {
                svfloat32_t result0 = zero;
                svfloat32_t result1 = zero;
                svfloat32_t result2 = zero;
                svfloat32_t result3 = zero;
                for (int64_t h = 0; h < 64; ++h) {
                    const float *scores = page_scores + h * 64;
                    const svfloat32_t weight = svdup_f32(row_weights[h]);
                    const svfloat32_t score0 = svld1_f32(all, scores + 0 * 16);
                    const svfloat32_t score1 = svld1_f32(all, scores + 1 * 16);
                    const svfloat32_t score2 = svld1_f32(all, scores + 2 * 16);
                    const svfloat32_t score3 = svld1_f32(all, scores + 3 * 16);
                    result0 = svmla_f32_x(
                        all, result0, svmax_f32_x(all, score0, zero), weight);
                    result1 = svmla_f32_x(
                        all, result1, svmax_f32_x(all, score1, zero), weight);
                    result2 = svmla_f32_x(
                        all, result2, svmax_f32_x(all, score2, zero), weight);
                    result3 = svmla_f32_x(
                        all, result3, svmax_f32_x(all, score3, zero), weight);
                }
                svst1_f32(all, out + 0 * 16, result0);
                svst1_f32(all, out + 1 * 16, result1);
                svst1_f32(all, out + 2 * 16, result2);
                const svbool_t last_pg = svwhilelt_b32(
                    uint64_t(48), uint64_t(valid_tokens));
                svst1_f32(last_pg, out + 3 * 16, result3);
            } else {
                for (int64_t tb = 0; tb < token_tiles; ++tb) {
                    svfloat32_t result = zero;
                    for (int64_t h = 0; h < 64; ++h) {
                        const svfloat32_t score = svld1_f32(
                            all, page_scores + h * 64 + tb * 16);
                        const svfloat32_t activated = svmax_f32_x(all, score, zero);
                        result = svmla_n_f32_x(
                            all, result, activated, row_weights[h]);
                    }
                    const svbool_t store_pg = svwhilelt_b32(
                        uint64_t(0), uint64_t(valid_tokens - tb * 16));
                    svst1_f32(store_pg, out + tb * 16, result);
                }
            }
#ifdef EN_TIMING
            _t_p += get_clock_us() - _t_phase;
#endif
        }
        ++logical_block;
    }

#ifdef EN_TIMING
    if (ctx->phase_timing && tid < omp_get_max_threads()) {
        ctx->phase_timing[tid].q_pack_us += _t_q;
        ctx->phase_timing[tid].k_pack_us += _t_k;
        ctx->phase_timing[tid].sme_us += _t_s;
        ctx->phase_timing[tid].postprocess_us += _t_p;
        ctx->phase_timing[tid].mask_us += _t_m;
        ctx->phase_timing[tid].total_us += get_clock_us() - _t_thread_start;
    }
#endif
}

} // namespace kupl_mqa_detail

#endif // __aarch64__

inline void kupl_bf16_paged_mqa_logits(
    const Tensor<bfloat16_t, 4> &q,          // [batch_size, next_n, num_heads, dim]
    const Tensor<bfloat16_t, 4> &kv_cache,   // [num_blocks, block_size, 1, dim]
    const Tensor<int64_t, 2> &block_tables,  // [batch_size, max_num_blocks]
    const Tensor<int64_t, 1> &context_lens,  // [batch_size]
    const Tensor<float, 2> &weights,         // [batch_size * next_n, num_heads]
    const Tensor<float, 2> &output,          // [batch_size * next_n, max_model_len] (output)
    int64_t batch_size,
    int64_t next_n,
    int64_t num_heads,
    int64_t dim,
    int64_t block_size,
    int64_t max_model_len,
    MqaLogitsPhaseTiming *phase_timing = nullptr
) {
#ifdef __aarch64__
    const auto *q_ptr = q.data_ptr();
    const auto *kv_ptr = kv_cache.data_ptr();
    const auto *block_table_ptr = block_tables.data_ptr();
    const auto *context_len_ptr = context_lens.data_ptr();
    const auto *weight_ptr = weights.data_ptr();
    auto *output_ptr = output.data_ptr();
    const int64_t max_num_blocks = block_tables.size(1);

    FLASH_ASSERT(block_size == 64 && num_heads == 64 && dim == 128);
    FLASH_ASSERT(next_n > 0 && next_n <= 2);

    const int64_t packed_q_stride = num_heads * dim;
    const int64_t packed_q_elements =
        batch_size * next_n * packed_q_stride;

    // 与 indexer 版相同的 NUMA 绑定共享 Workspace，跨调用复用。
    static thread_local PacMqaPackedQWorkspace packed_q_workspace;
    const int64_t packed_q_bytes =
        packed_q_elements * static_cast<int64_t>(sizeof(bfloat16_t));
    auto *packed_q_all = reinterpret_cast<bfloat16_t *>(
        packed_q_workspace.reserve(packed_q_bytes));

    // Page 前缀和：每个 worker 取一段连续 Page，内部只做一次 upper_bound 定位。
    static thread_local std::vector<int64_t> page_offsets_storage;
    page_offsets_storage.resize(static_cast<size_t>(batch_size + 1));
    int64_t *page_offsets = page_offsets_storage.data();
    page_offsets[0] = 0;
    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t context_len = context_len_ptr[b];
        FLASH_ASSERT(context_len >= next_n && context_len <= max_model_len);
        const int64_t num_blocks = ceil_div(context_len, block_size);
        FLASH_ASSERT(num_blocks <= max_num_blocks);
        page_offsets[b + 1] = page_offsets[b] + num_blocks;
    }
    const int64_t total_pages = page_offsets[batch_size];
    if (total_pages == 0) {
        return;
    }

    kupl_mqa_detail::KuplMqaPageCtx ctx;
    ctx.q = q_ptr;
    ctx.kv_cache = kv_ptr;
    ctx.block_tables = block_table_ptr;
    ctx.context_lens = context_len_ptr;
    ctx.weights = weight_ptr;
    ctx.output = output_ptr;
    ctx.batch_size = batch_size;
    ctx.next_n = next_n;
    ctx.num_heads = num_heads;
    ctx.dim = dim;
    ctx.block_size = block_size;
    ctx.max_model_len = max_model_len;
    ctx.max_num_blocks = max_num_blocks;
    ctx.packed_q_stride = packed_q_stride;
    ctx.packed_q_all = packed_q_all;
    ctx.page_offsets = page_offsets;
    ctx.total_pages = total_pages;
    ctx.phase_timing = phase_timing;

    // 使用 KUPL 默认全局 egroup（其执行器为 0..avail_pu_cnt，与探针显式创建的
    // egroup 等价），避免自行创建。worker 数取 kupl 执行器数与 OMP 线程数的
    // 较小值，保证与原版同核数对比，也避免无 taskset 时执行器数失控。
    const int kupl_executors = kupl_get_num_executors();
    if (kupl_executors <= 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                "[kupl_mqa] KUPL unavailable (executors=%d); falling back to OpenMP kernel.\n",
                kupl_executors);
        }
        indexer_bf16_paged_mqa_logits(
            q, kv_cache, block_tables, context_lens, weights, output,
            batch_size, next_n, num_heads, dim, block_size, max_model_len,
            phase_timing);
        return;
    }
    int workers = kupl_executors;
    const int omp_threads = omp_get_max_threads();
    if (omp_threads > 0 && omp_threads < workers) {
        workers = omp_threads;
    }
    {
        static bool printed = false;
        if (!printed) {
            printed = true;
            std::fprintf(stderr,
                "[kupl_mqa] using %d KUPL workers (kupl_executors=%d, omp_threads=%d).\n",
                workers, kupl_executors, omp_threads);
        }
    }

    // 单个 STATIC parallel_for：worker 收到连续 Page 区间 [lower, upper)。
    // blocksize 显式置 1（STATIC 策略下 kupl_check_range 也会归一化为 1）。
    kupl_nd_range_t range;
    KUPL_STRIDE_1D_RANGE_INIT(range, 0, total_pages, 1, 1);

    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = nullptr,   // 使用 KUPL 默认全局 egroup（全部执行器）
        .concurrency = workers,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };

    const int ret = kupl_parallel_for(&desc, kupl_mqa_detail::kupl_mqa_page_worker, &ctx);
    if (ret != KUPL_OK) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                "[kupl_mqa] kupl_parallel_for failed (%d); falling back to OpenMP kernel.\n",
                ret);
        }
        indexer_bf16_paged_mqa_logits(
            q, kv_cache, block_tables, context_lens, weights, output,
            batch_size, next_n, num_heads, dim, block_size, max_model_len,
            phase_timing);
    }
#else
    (void)phase_timing;
    ref_bf16_paged_mqa_logits<float>(
        q, kv_cache, block_tables, context_lens, weights, output,
        batch_size, next_n, num_heads, dim, block_size, max_model_len);
#endif
}
