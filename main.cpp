#include <iomanip>
#include <cstring>
#include <fstream>
#include <vector>
#include <numeric>
#include <iostream>
#include <random>

#include "Tensor.h"
#include "utils.h"
#include "allocator.h"
#include "testcase.h"
#include "ref_mqa_logits.h"
#include "indexer_mqa_logits.h"

constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t MAX_MODEL_LEN = 8192;
constexpr int64_t HEADS = 64;
constexpr int64_t DIM = 128;

constexpr int64_t MAX_ALLOC_SIZE = 2000LL * 1024 * 1024;

std::vector<TestCaseParams> gen_testcase()
{
    std::vector<TestCaseParams> testcases;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> params_list = {
        {32, 2, 1024},
        {128, 1, 4096}
    };
    for (auto [batch_size, next_n, avg_kv] : params_list) {
        TestCaseParams p;
        p.batch_size = batch_size;
        p.next_n = next_n;
        p.num_heads = HEADS;
        p.dim = DIM;
        p.block_size = BLOCK_SIZE;
        p.max_model_len = MAX_MODEL_LEN;
        p.check_correctness = true;
        p.num_runs = 100;
        p.avg_kv_len = avg_kv;

        testcases.push_back(p);
    }
    return testcases;
}

void run_test(const TestCaseParams &p, int64_t rand_seed = 0)
{
    static int64_t testcase_count = 0;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Running on TestCase " << (++testcase_count) << ":" << std::endl;
    std::cout << "Parameters: batch_size = " << p.batch_size
              << ", next_n = " << p.next_n
              << ", avg_kv = " << p.avg_kv_len << std::endl;

    std::mt19937 rng(rand_seed);
    Allocator allocator(MAX_ALLOC_SIZE);
    auto datalist = generate_datalist(p, allocator, rng);

    double total_time_usage_us = 0;
    double total_context_len = 0;
    for (int64_t i = 0; i < p.num_runs; ++i) {
        auto &d = datalist[i % datalist.size()];
        auto begin = get_clock_us();
        indexer_bf16_paged_mqa_logits(d.q, d.kv_cache, d.block_tables, d.context_lens,
            d.weights, d.logits, p.batch_size, p.next_n, p.num_heads, p.dim, p.block_size, p.max_model_len);
        auto end = get_clock_us();
        if (i > 0) {
            total_time_usage_us += end - begin;
            total_context_len += d.total_context_len;
        }
    }

    if (p.check_correctness) {
        auto &d = datalist[rng() % datalist.size()];
        ref_bf16_paged_mqa_logits<double>(d.q, d.kv_cache, d.block_tables, d.context_lens,
            d.weights, d.ref_logits, p.batch_size, p.next_n, p.num_heads, p.dim, p.block_size, p.max_model_len);
        FLASH_ASSERT(check_mask_and_replace(d.logits, d.ref_logits));
        auto cos_diff = calc_cos_diff(d.logits, d.ref_logits);
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "cos_diff: " << cos_diff << std::endl;
        FLASH_ASSERT(cos_diff < 5e-6);
    }

    if (p.num_runs > 1) {
        total_context_len /= p.num_runs - 1;
        double flops = 2.0 * total_context_len * p.next_n * p.num_heads * p.dim;
        double avg_time_usage_us = total_time_usage_us / (p.num_runs - 1);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Performance: " << (flops / avg_time_usage_us) / (1e12 / 1e6) << " TFLOPS" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    auto test_cases = gen_testcase();
    std::cout << "Total Number of Testcases: " << test_cases.size() << std::endl;

    for (const auto &p : test_cases) {
        run_test(p);
    }

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "All Tests Ends" << std::endl;

    return 0;
}