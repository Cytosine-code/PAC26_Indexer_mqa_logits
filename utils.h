#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <numeric>
#include <limits>
#include <time.h>
#include <iostream>
#include <random>

#ifdef __aarch64__
#include <arm_sme.h>
#include <arm_neon.h>
#else
// On x86, bfloat16_t is not a native type — define it as a 16-bit struct.
// All arithmetic happens in float; this type is only used for memory storage.
struct bfloat16_t {
    uint16_t raw;
};
#endif

#include "Tensor.h"

#define FLASH_ASSERT(cond)                                                      \
do {                                                                            \
    if (__builtin_expect(!(cond), 0)) {                                          \
        std::cerr << "Assertion Failed (" << __FILE__ << ":" << __LINE__ << "): " << #cond << std::endl; \
        exit(1);                                                                 \
    }                                                                            \
} while (0)

template <typename T>
constexpr T ceil_div(const T &x, const T &y)
{
    return (x + y - 1) / y;
}

template <typename T>
constexpr T ceil(const T &x, const T &y)
{
    return ceil_div(x, y) * y;
}

inline auto get_clock_us()
{
    struct timespec nowtime;
    clock_gettime(CLOCK_MONOTONIC, &nowtime);
    return nowtime.tv_sec * 1e6 + nowtime.tv_nsec * 1e-3;
}

#ifdef __aarch64__

static inline bfloat16_t to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

static inline float to_float(bfloat16_t x)
{
    return vcvtah_f32_bf16(x);
}

#else

// Software bfloat16 <-> float32 conversion (round-to-nearest-even for FP32→BF16)
static inline bfloat16_t to_bf16(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    // Round to nearest even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    bfloat16_t result;
    result.raw = static_cast<uint16_t>(bits >> 16);
    return result;
}

static inline float to_float(bfloat16_t x)
{
    uint32_t bits = static_cast<uint32_t>(x.raw) << 16;
    float result;
    memcpy(&result, &bits, sizeof(bits));
    return result;
}

#endif