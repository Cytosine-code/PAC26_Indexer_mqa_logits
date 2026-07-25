## 计算节点 SVE / SME 软硬件配置汇总

以下结论来自 `compute_env` 输出和实际编译、反汇编、运行探针，不只是依据 CPU flags 推断。

### 硬件环境

| 项目 | 实测配置 |
|---|---|
| 架构 | AArch64 |
| CPU 厂商 | HiSilicon |
| CPU part | `0xd22` |
| CPU 数量 | 608 核 |
| 拓扑 | 2 sockets × 304 cores |
| SMT | 无，1 thread/core |
| 主频 | 最高 2.0 GHz |
| NUMA | 32 个节点，每节点约 38 核 |
| L1 Data | 32 KiB/core |
| L1 Instruction | 32 KiB/core |
| L2 | 768 KiB/core |
| Cache line | 64 bytes |

CPU flags 中已确认：

```text
sve sve2 svebf16
bf16
sme smeb16f32
```

此外还有：

```text
svei8mm
i8mm
asimdbf16
smef16f32
smef32f32
smef64f64
smei8i32
```

### SVE 配置

| 项目 | 实测结果 |
|---|---:|
| SVE VL | 512 bit |
| SVE VL | 64 bytes |
| BF16 lanes/vector | 32 |
| FP32 lanes/vector | 16 |
| `PR_SVE_GET_VL` | 64 bytes |
| `svcntb()` | 64 |
| `svcnth()` | 32 |
| SVE2 | 支持 |
| SVE BF16 `BFDOT` | 支持并通过运行测试 |

编译器宏：

```text
__ARM_FEATURE_SVE=1
__ARM_FEATURE_SVE_BITS=0
__ARM_FEATURE_SVE_BF16=undefined
__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=1
```

`__ARM_FEATURE_SVE_BITS=0` 表示编译器使用长度无关的 SVE 编程模型，不代表向量长度为 0。运行时真实 VL 是 512 bit。

需要特别注意：

```text
__ARM_FEATURE_SVE_BF16 未定义
```

但这只是厂商 GCC 特性宏不完整，并不代表 SVE BF16 不可用。实际已经验证：

```cpp
svbfdot_f32(acc, a, b)
```

能够编译，反汇编生成：

```asm
bfdot z0.s, z1.h, z2.h
```

运行测试结果：

```text
reference=2.15625
sve=2.15625
abs_error=0
PASS: SVE BF16 dot
```

因此正式代码不要使用以下宏判断 SVE BF16 路径：

```cpp
#ifdef __ARM_FEATURE_SVE_BF16
```

在比赛已知编译参数和硬件环境下，可以在 `__aarch64__` 路径直接使用 `svbfdot_f32`。

### SVE BF16 指令语义

一条 512-bit `BFDOT`：

- 输入：两个各含 32 个 BF16 的 Z 寄存器；
- 输出累加：16 个 FP32 lane；
- 每个 FP32 lane 累加两个 BF16 乘积；
- 总计执行 32 次 BF16 乘法和 32 次 FP32 累加；
- 按 MAC 计为 64 FLOPs。

对于长度为 128 的单个 Q·K 点积：

```text
128 BF16 / 32 BF16 per vector = 4 次向量处理
```

因此需要：

```text
4 次 Q load
4 次 K load
4 条 BFDOT
1 次 FP32 horizontal reduction
```

### SME 配置

| 项目 | 实测结果 |
|---|---:|
| SME | 支持 |
| SME SVL | 512 bit |
| SME SVL | 64 bytes |
| `PR_SME_GET_VL` | 64 bytes |
| `arm_sme.h` | 存在 |
| `__ARM_FEATURE_SME` | 1 |
| ZA storage | 支持并通过测试 |
| BF16→FP32 BFMOPA | 支持并通过测试 |
| Streaming mode | 支持并通过测试 |

CPU 能力标志包含：

```text
sme
smeb16f32
```

其中 `smeb16f32` 表示支持 BF16 输入、FP32 累加的 SME outer-product。

实际生成并成功运行的核心指令包括：

```asm
smstart
zero   {za}
bfmopa za0.s, p0/m, p0/m, z0.h, z1.h
st1w   {za0h.s[...]}, p2, [...]
smstop
```

### SME tile 形状

SME SVL 为 512 bit，即 64 bytes。对 FP32 tile：

```text
64 bytes / 4 bytes per FP32 = 16
```

因此：

```text
ZA0.S tile = 16 rows × 16 columns FP32
```

单个 tile 占用：

```text
16 × 16 × 4 = 1024 bytes
```

ZA 可以按四个 FP32 tile 视图访问：

```text
ZA0.S
ZA1.S
ZA2.S
ZA3.S
```

但当前探针已实际验证的是 `ZA0.S`。

### BFMOPA 计算语义

核心指令：

```asm
bfmopa za0.s, p0/m, p0/m, z0.h, z1.h
```

对于 512-bit SVL：

- `z0.h`：32 个 BF16；
- `z1.h`：32 个 BF16；
- 每两个 BF16 构成 K 维的一组；
- 左源表示 `16×2` BF16 子矩阵；
- 右源表示 `2×16` BF16 子矩阵；
- 一条指令计算并累加一个 `16×16` FP32 outer-product tile；
- K 维每条指令推进 2。

即：

```text
ZA[16,16] += A[16,2] × B[2,16]
```

每条 BFMOPA 完成：

```text
16 × 16 × 2 = 512 MAC
                    = 1024 FLOPs
```

算子固定 `dim=128`，所以一个完整 `16×16` tile 需要：

```text
128 / 2 = 64 条 BFMOPA
```

每个 tile 总计算量：

```text
64 × 1024 = 65,536 FLOPs
```

### BFMOPA packing 要求

实测正确的数据布局如下。

左源 `Zn` 按输出行存放相邻 K-pair：

```text
A(row0,k0), A(row0,k1),
A(row1,k0), A(row1,k1),
...
A(row15,k0), A(row15,k1)
```

右源 `Zm` 同样按输出列存放相邻 K-pair：

```text
B(k0,col0), B(k1,col0),
B(k0,col1), B(k1,col1),
...
B(k0,col15), B(k1,col15)
```

因此，矩阵乘中左右两个输入都需要围绕 K-pair 做交错 packing。

对当前算子：

- 左源对应 16 个 query heads；
- 右源对应 16 个 KV tokens；
- 每次 packing 两个连续 dim；
- 每条 BFMOPA 将这两个 dim 对整个 `16×16` 输出 tile 的贡献一次算完。

### BFMOPA 谓词要求

这是探针调试中确认的重要细节。

BFMOPA 输入寄存器使用 `.h`，谓词也必须按 halfword 粒度生成：

```asm
ptrue p0.h
bfmopa za0.s, p0/m, p0/m, z0.h, z1.h
```

不能使用：

```asm
ptrue p0.s
```

如果用 `.s` 粒度，BF16 pair 中只有一半元素会被激活，最终每个输出只包含 K-pair 的第一项乘积。

ZA 中存放的是 FP32，所以写回谓词应按 `.s` 粒度单独生成：

```asm
ptrue p2.s
st1w {za0h.s[...]}, p2, [...]
```

推荐分别维护：

```asm
p0.h    // BFMOPA 输入
p1.h    // BF16 load
p2.s    // FP32 ZA store
```

### Streaming mode 与 ZA

`BFMOPA` 需要同时启用 streaming mode 和 ZA：

```asm
smstart
```

无参数 `smstart` 会同时设置：

```text
PSTATE.SM = 1
PSTATE.ZA = 1
```

只使用：

```asm
smstart za
```

是不够的，因为它只启用 ZA，不启用 streaming mode。

计算完成后：

```asm
smstop
```

同时退出 streaming mode 并关闭 ZA。

需要注意 streaming mode 的向量长度使用 SME SVL，理论上可能与普通 SVE VL 不同。本节点两者恰好都是：

```text
512 bit / 64 bytes
```

### 编译环境

| 项目 | 配置 |
|---|---|
| OS/toolchain | openEuler 3.0.3 工具链 |
| GCC | 12.3.1 |
| C++ compiler | `g++` |
| ARM headers | `arm_sve.h`、`arm_sme.h`、`arm_bf16.h` 可用 |
| GNU assembler | 能识别 SME、BFMOPA 和 ZA 指令 |

比赛使用的编译命令：

```bash
g++ -O3 \
    -march=armv9-a+sme+sve2 \
    -fopenmp \
    -lnuma \
    main.cpp \
    -o main
```

该命令已经能够：

- 编译 SVE BF16 intrinsic；
- 汇编 SME inline assembly；
- 生成并运行 `BFDOT`；
- 生成并运行 `BFMOPA`。

但 GCC 12.3.1 的 SME ACLE 支持程度不宜按新版 GCC 假定。当前最可靠的策略是：

- 普通 SVE 运算使用 intrinsic；
- SVE BF16 `svbfdot_f32` 已验证可直接使用；
- SME BFMOPA/ZA 操作使用已验证的 GNU inline/global assembly；
- 不依赖 `__ARM_FEATURE_SVE_BF16`；
- 不依赖新版 SME function attributes 或新版 `arm_sme.h` intrinsic 名称。

### 对当前算子的直接映射

固定算子规模：

```text
heads = 64
tokens/page = 64
dim = 128
```

SME tile：

```text
16 heads × 16 tokens
```

一个 page/query 需要：

```text
heads tiles  = 64 / 16 = 4
token tiles  = 64 / 16 = 4
tile 总数    = 4 × 4 = 16
```

每个 tile：

```text
128 / 2 = 64 条 BFMOPA
```

因此一个完整 page/query：

```text
16 × 64 = 1024 条 BFMOPA
```

对应计算量：

```text
1024 × 1024 FLOPs = 1,048,576 FLOPs
```

恰好等于：

```text
2 × 64 heads × 64 tokens × 128 dim
= 1,048,576 FLOPs
```

结论是，该节点的 512-bit SVE/SME 与题目的 `64×64×128` page 计算形状高度匹配：每个 page 可精确拆成 16 个 `16×16` SME tile，K 维精确拆成 64 次 BF16 pair outer product，不需要处理维度尾部。