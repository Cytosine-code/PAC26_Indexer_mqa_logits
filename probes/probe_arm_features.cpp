#include <cstdio>
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <arm_sve.h>

#if defined(__has_include)
#if __has_include(<arm_sme.h>)
#include <arm_sme.h>
#define PAC_HAS_ARM_SME_HEADER 1
#else
#define PAC_HAS_ARM_SME_HEADER 0
#endif
#else
#define PAC_HAS_ARM_SME_HEADER 0
#endif

#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif
#ifndef PR_SME_VL_LEN_MASK
#define PR_SME_VL_LEN_MASK 0xffff
#endif
#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

int main()
{
    std::printf("compiler: %s\n", __VERSION__);
    std::printf("arm_sme.h: %s\n", PAC_HAS_ARM_SME_HEADER ? "yes" : "no");

#ifdef __ARM_FEATURE_SVE
    std::printf("__ARM_FEATURE_SVE=%d\n", __ARM_FEATURE_SVE);
#else
    std::printf("__ARM_FEATURE_SVE=undefined\n");
#endif
#ifdef __ARM_FEATURE_SVE_BITS
    std::printf("__ARM_FEATURE_SVE_BITS=%d\n", __ARM_FEATURE_SVE_BITS);
#else
    std::printf("__ARM_FEATURE_SVE_BITS=undefined (length-agnostic build)\n");
#endif
#ifdef __ARM_FEATURE_SVE_BF16
    std::printf("__ARM_FEATURE_SVE_BF16=%d\n", __ARM_FEATURE_SVE_BF16);
#else
    std::printf("__ARM_FEATURE_SVE_BF16=undefined\n");
#endif
#ifdef __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    std::printf("__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=%d\n",
                __ARM_FEATURE_BF16_VECTOR_ARITHMETIC);
#else
    std::printf("__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=undefined\n");
#endif
#ifdef __ARM_FEATURE_SME
    std::printf("__ARM_FEATURE_SME=%d\n", __ARM_FEATURE_SME);
#else
    std::printf("__ARM_FEATURE_SME=undefined\n");
#endif

    const int sve_vl = prctl(PR_SVE_GET_VL);
    const int sme_vl = prctl(PR_SME_GET_VL);
    std::printf("svcntb()=%zu bytes\n", svcntb());
    std::printf("PR_SVE_GET_VL=%d bytes\n",
                sve_vl < 0 ? sve_vl : (sve_vl & PR_SVE_VL_LEN_MASK));
    std::printf("PR_SME_GET_VL=%d bytes\n",
                sme_vl < 0 ? sme_vl : (sme_vl & PR_SME_VL_LEN_MASK));
    return 0;
}
