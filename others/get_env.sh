# ========== 1. CPU 基本信息 ==========
lscpu                             # 架构、核心数、频率、NUMA
cat /proc/cpuinfo | head -80      # CPU model、implementer、flags

# ========== 2. SVE / SME 向量宽度（最关键！）==========
cat /proc/sys/arch/vector_length  # 如果有的话
# 或者写个微型程序查看：
cat > /tmp/check_sve.c << 'EOF'
#include <stdio.h>
#include <arm_sve.h>
int main() {
    printf("SVE VL (bits): %ld\n", svcntb() * 8);
    printf("SVE VL (bytes): %ld\n", svcntb());
    return 0;
}
EOF
g++ -O0 -march=armv9-a+sve /tmp/check_sve.c -o /tmp/check_sve
/tmp/check_sve

# ========== 3. SME 支持状态 ==========
# 同样写个小程序查 SME 向量长度（不一定有 SME 暴露给用户态）
cat > /tmp/check_sme.c << 'EOF'
#include <stdio.h>
#include <arm_sme.h>
int main() {
    printf("SME SVL (bits): %ld\n", svcntsw() * 8);
    return 0;
}
EOF
g++ -O0 -march=armv9-a+sme /tmp/check_sme.c -o /tmp/check_sme
/tmp/check_sme

# ========== 4. 缓存层次（手写 tiling 时需要）==========
lscpu | grep -i cache
# 或
cat /sys/devices/system/cpu/cpu0/cache/index*/size
cat /sys/devices/system/cpu/cpu0/cache/index*/type
cat /sys/devices/system/cpu/cpu0/cache/index*/coherency_line_size

# ========== 5. NUMA 拓扑（确认单节点）==========
numactl --hardware
numactl --show
ls /sys/devices/system/node/

# ========== 6. 内存带宽参考 ==========
# 如果有 dmidecode 权限
sudo dmidecode -t memory 2>/dev/null | grep -E "Speed|Type|Size"

# ========== 7. 编译器 & 版本 ==========
g++ --version
clang++ --version
# 检查编译器是否支持 SME intrinsic
echo "#include <arm_sme.h>" | g++ -march=armv9-a+sme -x c++ -c - -o /dev/null 2>&1
