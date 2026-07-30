#pragma once

#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>

#include "ref_mqa_logits.h"
#include "allocator.h"
#include "Tensor.h"
#include "utils.h"

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#endif

// Per-thread timing accumulators for phased profiling.
// Pass an array (size = omp_get_max_threads()) to indexer_bf16_paged_mqa_logits.
// Times are in microseconds, summed over all rows/pages processed by one worker.
struct MqaLogitsPhaseTiming {
    double q_pack_us = 0;
    double k_pack_us = 0;
    double sme_us = 0;
    double postprocess_us = 0;
    double mask_us = 0;
    double total_us = 0;       // wall clock of the entire batch loop body
};

#ifdef __aarch64__
extern "C" void pac_sme_page_scores(
    const bfloat16_t *packed_q, const bfloat16_t *packed_k,
    float *scores, int64_t token_tiles);
extern "C" void pac_sme_page_scores_2x2(
    const bfloat16_t *packed_q, const bfloat16_t *packed_k,
    float *scores);
extern "C" void pac_sme_pack_k_page(
    const bfloat16_t *source, bfloat16_t *packed);
extern "C" void pac_sme_pack_k_page_and_scores(
    const bfloat16_t *source, bfloat16_t *packed,
    const bfloat16_t *packed_q, float *scores, int64_t token_tiles);
extern "C" void pac_sme_pack_k_page_and_scores_2x2(
    const bfloat16_t *source, bfloat16_t *packed,
    const bfloat16_t *packed_q, float *scores);

asm(R"(
    .arch armv9-a+sme+sve2
    .text
    .align 4
    .global pac_sme_page_scores
    .type pac_sme_page_scores, %function
pac_sme_page_scores:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
.Lpac_scores_streaming:
    ptrue p0.h
    ptrue p1.h
    ptrue p2.s
    mov x7, x2

1:
    zero {za}
    mov x4, x0
    mov x5, x1
    mov x6, #64

2:
    ld1h {z0.h}, p1/z, [x4]
    ld1h {z1.h}, p1/z, [x4, #1, mul vl]
    ld1h {z2.h}, p1/z, [x4, #2, mul vl]
    ld1h {z3.h}, p1/z, [x4, #3, mul vl]
    ld1h {z4.h}, p1/z, [x5]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za2.s, p0/m, p0/m, z2.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z3.h, z4.h
    add x4, x4, #256
    add x5, x5, #64
    subs x6, x6, #1
    b.ne 2b

    mov w12, #0
    mov x8, x7
3:
    add x9, x8, #1, lsl #12
    add x10, x8, #2, lsl #12
    add x11, x8, #3, lsl #12
    st1w {za0h.s[w12, 0]}, p2, [x8]
    st1w {za1h.s[w12, 0]}, p2, [x9]
    st1w {za2h.s[w12, 0]}, p2, [x10]
    st1w {za3h.s[w12, 0]}, p2, [x11]
    add x8, x8, #256
    add w12, w12, #1
    cmp w12, #16
    b.lo 3b

    add x1, x1, #1, lsl #12
    add x7, x7, #64
    subs x3, x3, #1
    b.ne 1b
    smstop
    ldp d8, d9, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    ret
    .size pac_sme_page_scores, .-pac_sme_page_scores

    .align 4
    .global pac_sme_page_scores_2x2
    .type pac_sme_page_scores_2x2, %function
pac_sme_page_scores_2x2:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart

.Lpac_scores_2x2_streaming:
    ptrue p0.h
    ptrue p1.h
    ptrue p2.s
    mov x16, #2
    mov x17, x1

.Lpac_scores_2x2_heads:
    mov x13, #2
    mov x14, x17
    mov x15, x2

.Lpac_scores_2x2_tokens:
    zero {za}
    mov x4, x0
    mov x5, x14
    add x6, x14, #1, lsl #12
    mov x7, #64

.Lpac_scores_2x2_k:
    ld1h {z0.h}, p1/z, [x4]
    ld1h {z1.h}, p1/z, [x4, #1, mul vl]
    ld1h {z4.h}, p1/z, [x5]
    ld1h {z5.h}, p1/z, [x6]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    add x4, x4, #256
    add x5, x5, #64
    add x6, x6, #64
    subs x7, x7, #1
    b.ne .Lpac_scores_2x2_k

    mov x8, x15
    add x9, x8, #64
    add x10, x8, #1, lsl #12
    add x11, x10, #64
    mov w12, #0
.Lpac_scores_2x2_store:
    st1w {za0h.s[w12, 0]}, p2, [x8]
    st1w {za1h.s[w12, 0]}, p2, [x9]
    st1w {za2h.s[w12, 0]}, p2, [x10]
    st1w {za3h.s[w12, 0]}, p2, [x11]
    add x8, x8, #256
    add x9, x9, #256
    add x10, x10, #256
    add x11, x11, #256
    add w12, w12, #1
    cmp w12, #16
    b.lo .Lpac_scores_2x2_store

    add x14, x14, #2, lsl #12
    add x15, x15, #128
    subs x13, x13, #1
    b.ne .Lpac_scores_2x2_tokens

    add x0, x0, #128
    add x2, x2, #2, lsl #12
    subs x16, x16, #1
    b.ne .Lpac_scores_2x2_heads

    smstop
    ldp d8, d9, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    ret
    .size pac_sme_page_scores_2x2, .-pac_sme_page_scores_2x2

    .align 4
    .global pac_sme_pack_k_page
    .type pac_sme_pack_k_page, %function
pac_sme_pack_k_page:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    adr x10, .Lpac_pack_only_done
    b .Lpac_pack_streaming

.Lpac_pack_streaming:
    ptrue p0.s
    mov x2, #4
    mov x8, x0
    mov x9, x1

4:
    mov x3, #4
    mov x4, x8
    mov x6, x9
5:
    mov w12, #0
    mov x5, x4
6:
    ld1w {za0h.s[w12, 0]}, p0/z, [x5]
    add x5, x5, #256
    add w12, w12, #1
    cmp w12, #16
    b.lo 6b

    mov w12, #0
7:
    st1w {za0v.s[w12, 0]}, p0, [x6]
    add x6, x6, #64
    add w12, w12, #1
    cmp w12, #16
    b.lo 7b

    add x4, x4, #64
    subs x3, x3, #1
    b.ne 5b

    add x8, x8, #4096
    add x9, x9, #4096
    subs x2, x2, #1
    b.ne 4b
    br x10

.Lpac_pack_only_done:
    smstop
    ldp d8, d9, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    ret
    .size pac_sme_pack_k_page, .-pac_sme_pack_k_page

    .align 4
    .global pac_sme_pack_k_page_and_scores
    .type pac_sme_pack_k_page_and_scores, %function
pac_sme_pack_k_page_and_scores:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    mov x13, x2
    mov x14, x3
    mov x15, x4
    smstart
    adr x10, .Lpac_pack_and_scores_done
    b .Lpac_pack_streaming

.Lpac_pack_and_scores_done:
    mov x0, x13
    mov x2, x14
    mov x3, x15
    b .Lpac_scores_streaming
    .size pac_sme_pack_k_page_and_scores, .-pac_sme_pack_k_page_and_scores

    .align 4
    .global pac_sme_pack_k_page_and_scores_2x2
    .type pac_sme_pack_k_page_and_scores_2x2, %function
pac_sme_pack_k_page_and_scores_2x2:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    mov x13, x2
    mov x14, x3
    smstart
    adr x10, .Lpac_pack_and_scores_2x2_done
    b .Lpac_pack_streaming

.Lpac_pack_and_scores_2x2_done:
    mov x0, x13
    mov x2, x14
    b .Lpac_scores_2x2_streaming
    .size pac_sme_pack_k_page_and_scores_2x2, .-pac_sme_pack_k_page_and_scores_2x2
)");

struct PacMqaPackedQWorkspace {
    void *address = nullptr;
    int64_t capacity = 0;

    ~PacMqaPackedQWorkspace()
    {
        if (address) {
            munmap(address, capacity);
        }
    }

    void *reserve(int64_t requested_bytes)
    {
        const int64_t required = ceil(requested_bytes, PAGE_SIZE);
        if (required > capacity) {
            void *new_address = mmap_on_package_memory(required);
            FLASH_ASSERT(new_address != MAP_FAILED);
            if (address) {
                munmap(address, capacity);
            }
            address = new_address;
            capacity = required;
        }
        return address;
    }
};
#endif

inline void indexer_bf16_paged_mqa_logits(
    const Tensor<bfloat16_t, 4> &q,          // [batch_size, next_n, num_heads, dim]
    const Tensor<bfloat16_t, 4> &kv_cache,   // [num_blocks, block_size, 1, dim]
    const Tensor<int64_t, 2> &block_tables,  // [batch_size, max_num_blocks]
    const Tensor<int64_t, 1> &context_lens,   // [batch_size]
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
    const auto *q_ptr = q.data_ptr();
    const auto *kv_ptr = kv_cache.data_ptr();
    const auto *block_table_ptr = block_tables.data_ptr();
    const auto *context_len_ptr = context_lens.data_ptr();
    const auto *weight_ptr = weights.data_ptr();
    auto *output_ptr = output.data_ptr();
    const int64_t max_num_blocks = block_tables.size(1);

    FLASH_ASSERT(block_size == 64 && num_heads == 64 && dim == 128);
    FLASH_ASSERT(next_n > 0 && next_n <= 2);

#ifdef __aarch64__
    const int64_t packed_q_stride = num_heads * dim;
    const int64_t packed_q_elements =
        batch_size * next_n * packed_q_stride;

    // Reuse NUMA-bound storage across benchmark invocations. The first call
    // for each test shape grows it; subsequent timed calls only overwrite it.
    static thread_local PacMqaPackedQWorkspace packed_q_workspace;
    const int64_t packed_q_bytes =
        packed_q_elements * static_cast<int64_t>(sizeof(bfloat16_t));
    auto *packed_q_all = reinterpret_cast<bfloat16_t *>(
        packed_q_workspace.reserve(packed_q_bytes));

    // Prefix sums let each OpenMP worker take one contiguous range of valid
    // pages. A worker locates its first batch once, then advances linearly.
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

#pragma omp parallel
    {
        alignas(64) bfloat16_t packed_k[64 * 128];
        alignas(64) float page_scores[64 * 64];
        const svbool_t pack_pg = svptrue_b32();
        const svuint32_t pack_offsets = svindex_u32(0, 256);

#ifdef EN_TIMING
        double _t_q = 0, _t_k = 0, _t_s = 0, _t_p = 0, _t_m = 0;
        const double _t_thread_start = get_clock_us();
        double _t_phase;
#endif

        // Phase 1: materialize every packed Q once and initialize its mask.
        // The implicit barrier makes packed_q_all visible to page workers.
#pragma omp for schedule(static)
        for (int64_t row = 0; row < batch_size * next_n; ++row) {
            const int64_t batch_idx = row / next_n;
            const int64_t n = row - batch_idx * next_n;
            const int64_t context_len = context_len_ptr[batch_idx];
            const bfloat16_t *q_row =
                q_ptr + row * packed_q_stride;
            bfloat16_t *packed_q =
                packed_q_all + row * packed_q_stride;

#ifdef EN_TIMING
            _t_phase = get_clock_us();
#endif
            for (int64_t kp = 0; kp < 64; ++kp) {
                for (int64_t hb = 0; hb < 4; ++hb) {
                    bfloat16_t *dst = packed_q + (kp * 4 + hb) * 32;
                    const bfloat16_t *src =
                        q_row + hb * 16 * dim + kp * 2;
                    const svuint32_t pairs = svld1_gather_u32offset_u32(
                        pack_pg, reinterpret_cast<const uint32_t *>(src),
                        pack_offsets);
                    svst1_u32(
                        pack_pg, reinterpret_cast<uint32_t *>(dst), pairs);
                }
            }
#ifdef EN_TIMING
            _t_q += get_clock_us() - _t_phase;
            _t_phase = get_clock_us();
#endif
            float *out = output_ptr + row * max_model_len;
            const int64_t valid_length = context_len - next_n + n + 1;
            std::fill(out + valid_length, out + max_model_len, -INFINITY);
#ifdef EN_TIMING
            _t_m += get_clock_us() - _t_phase;
#endif
        }

        const int64_t tid = omp_get_thread_num();
        const int64_t thread_count = omp_get_num_threads();
        const int64_t page_begin = total_pages * tid / thread_count;
        const int64_t page_end = total_pages * (tid + 1) / thread_count;

        int64_t batch_idx = batch_size;
        int64_t logical_block = 0;
        if (page_begin < page_end) {
            batch_idx = static_cast<int64_t>(
                std::upper_bound(
                    page_offsets, page_offsets + batch_size + 1,
                    page_begin) - page_offsets - 1);
            logical_block = page_begin - page_offsets[batch_idx];
        }

        for (int64_t page_index = page_begin;
             page_index < page_end; ++page_index) {
            while (batch_idx < batch_size &&
                   logical_block >=
                       page_offsets[batch_idx + 1] - page_offsets[batch_idx]) {
                ++batch_idx;
                logical_block = 0;
            }

            const int64_t context_len = context_len_ptr[batch_idx];
            const int64_t physical_block = block_table_ptr[
                batch_idx * max_num_blocks + logical_block];
            const int64_t token_base = logical_block * block_size;

            if (physical_block < 0) {
                for (int64_t n = 0; n < next_n; ++n) {
                    const int64_t row = batch_idx * next_n + n;
                    const int64_t valid_length =
                        context_len - next_n + n + 1;
                    const int64_t invalid_tokens = std::min<int64_t>(
                        block_size, valid_length - token_base);
                    if (invalid_tokens > 0) {
                        float *out = output_ptr +
                            row * max_model_len + token_base;
                        std::fill(out, out + invalid_tokens, -INFINITY);
                    }
                }
                ++logical_block;
                continue;
            }

            const bfloat16_t *k_page =
                kv_ptr + physical_block * block_size * dim;

#ifdef EN_TIMING
            _t_phase = get_clock_us();
            pac_sme_pack_k_page(k_page, packed_k);
            _t_k += get_clock_us() - _t_phase;
#endif

            bool packed_k_ready = false;
            for (int64_t n = 0; n < next_n; ++n) {
                const int64_t row = batch_idx * next_n + n;
                const int64_t q_limit = context_len - next_n + n;
                const int64_t valid_tokens = std::min<int64_t>(
                    block_size, q_limit + 1 - token_base);
                if (valid_tokens <= 0) {
                    continue;
                }
                const int64_t token_tiles =
                    ceil_div(valid_tokens, int64_t(16));
                const bfloat16_t *packed_q =
                    packed_q_all + row * packed_q_stride;

#ifdef EN_TIMING
                _t_phase = get_clock_us();
                if (valid_tokens == 64) {
                    pac_sme_page_scores_2x2(
                        packed_q, packed_k, page_scores);
                } else {
                    pac_sme_page_scores(
                        packed_q, packed_k, page_scores, token_tiles);
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
                            k_page, packed_k, packed_q,
                            page_scores, token_tiles);
                    }
                    packed_k_ready = true;
                } else {
                    if (valid_tokens == 64) {
                        pac_sme_page_scores_2x2(
                            packed_q, packed_k, page_scores);
                    } else {
                        pac_sme_page_scores(
                            packed_q, packed_k,
                            page_scores, token_tiles);
                    }
                }
#endif
                const float *row_weights = weight_ptr + row * num_heads;
                float *out =
                    output_ptr + row * max_model_len + token_base;
                const svbool_t all = svptrue_b32();
                const svfloat32_t zero = svdup_f32(0.0f);
                if (token_tiles == 4) {
                    svfloat32_t result0 = zero;
                    svfloat32_t result1 = zero;
                    svfloat32_t result2 = zero;
                    svfloat32_t result3 = zero;
                    for (int64_t h = 0; h < 64; ++h) {
                        const float *scores = page_scores + h * 64;
                        const svfloat32_t weight =
                            svdup_f32(row_weights[h]);
                        const svfloat32_t score0 =
                            svld1_f32(all, scores + 0 * 16);
                        const svfloat32_t score1 =
                            svld1_f32(all, scores + 1 * 16);
                        const svfloat32_t score2 =
                            svld1_f32(all, scores + 2 * 16);
                        const svfloat32_t score3 =
                            svld1_f32(all, scores + 3 * 16);
                        result0 = svmla_f32_x(
                            all, result0,
                            svmax_f32_x(all, score0, zero), weight);
                        result1 = svmla_f32_x(
                            all, result1,
                            svmax_f32_x(all, score1, zero), weight);
                        result2 = svmla_f32_x(
                            all, result2,
                            svmax_f32_x(all, score2, zero), weight);
                        result3 = svmla_f32_x(
                            all, result3,
                            svmax_f32_x(all, score3, zero), weight);
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
                            const svfloat32_t activated =
                                svmax_f32_x(all, score, zero);
                            result = svmla_n_f32_x(
                                all, result, activated, row_weights[h]);
                        }
                        const svbool_t store_pg = svwhilelt_b32(
                            uint64_t(0),
                            uint64_t(valid_tokens - tb * 16));
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
        if (phase_timing) {
            phase_timing[tid].q_pack_us += _t_q;
            phase_timing[tid].k_pack_us += _t_k;
            phase_timing[tid].sme_us += _t_s;
            phase_timing[tid].postprocess_us += _t_p;
            phase_timing[tid].mask_us += _t_m;
            phase_timing[tid].total_us +=
                get_clock_us() - _t_thread_start;
        }
#endif
    }
#else
#pragma omp parallel for schedule(static)
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t context_len = context_len_ptr[batch_idx];
        const int64_t num_blocks = ceil_div(context_len, block_size);

        for (int64_t n = 0; n < next_n; ++n) {
            const int64_t row = batch_idx * next_n + n;
            const int64_t q_limit = context_len - next_n + n;
            float *out = output_ptr + row * max_model_len;

            // Preserve the reference contract: every masked position is -inf.
            std::fill(out, out + max_model_len, -INFINITY);

            const bfloat16_t *q_row =
                q_ptr + row * num_heads * dim;
            const float *row_weights = weight_ptr + row * num_heads;

            for (int64_t logical_block = 0; logical_block < num_blocks; ++logical_block) {
                const int64_t token_base = logical_block * block_size;
                const int64_t valid_tokens = std::min<int64_t>(
                    block_size, q_limit + 1 - token_base);
                if (valid_tokens <= 0) {
                    break;
                }

                const int64_t physical_block =
                    block_table_ptr[batch_idx * max_num_blocks + logical_block];
                if (physical_block < 0) {
                    continue;
                }
                const bfloat16_t *k_page =
                    kv_ptr + physical_block * block_size * dim;

                alignas(64) float page_output[64] = {};

                // One 64-token KV page remains cache-resident while all heads
                // consume it. ReLU, weight and head reduction are fused into
                // page_output, so there is no per-head scores allocation.
                for (int64_t h = 0; h < num_heads; ++h) {
                    const bfloat16_t *q_head = q_row + h * dim;
                    const float weight = row_weights[h];

                    for (int64_t t = 0; t < valid_tokens; ++t) {
                        const bfloat16_t *k_token = k_page + t * dim;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < dim; ++d) {
                            dot += to_float(q_head[d]) * to_float(k_token[d]);
                        }
                        page_output[t] += std::max(0.0f, dot) * weight;
                    }
                }

                std::copy(page_output, page_output + valid_tokens, out + token_base);
            }
        }
    }
#endif
}
