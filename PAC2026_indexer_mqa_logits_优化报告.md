# PAC 2026 `indexer_mqa_logits` 算子优化报告

## 1. 赛题分析

### 1.1 算法功能

赛题要求在鲲鹏处理器上优化 `indexer_bf16_paged_mqa_logits`。算子输入包括 BF16 格式的 Query、分页存储的 KV Cache、逻辑页到物理页的映射表、上下文长度以及各 Query Head 的权重，输出每个 Query 对历史 Token 的加权 Logits。

忽略分页寻址后，对一个 batch 中的一个 query，核心计算可以表示为：

```text
score[h,t] = dot(Q[h,:], K[t,:])
output[t]  = sum_h weight[h] * ReLU(score[h,t])
```

其中：

- Query Head 数为 64；
- Head Dimension 为 128；
- KV Page 大小为 64 Token；
- KV Cache 使用 BF16 存储；
- 点积采用 FP32 累加；
- 超过当前 Query 因果位置的输出必须写为负无穷；
- KV 的逻辑页通过 `block_tables` 映射到物理页。

完整计算过程包含以下步骤：

1. 根据 `block_tables` 找到当前 batch 的物理 KV Page；
2. 计算 64 个 Query Head 与有效历史 Token 的 128 维点积；
3. 对每个点积结果执行 ReLU；
4. 乘以对应 Head 的权重；
5. 沿 Head 维归约，得到最终 Token Logits；
6. 对因果 Mask 之外的位置写入 `-INFINITY`。

### 1.2 固定测试规模与计算量

测试包含两个固定用例：

| 测试用例 | Batch Size | Next N | 平均 KV 长度 | Heads | Dim | Page Size |
|---|---:|---:|---:|---:|---:|---:|
| Case 1 | 32 | 2 | 1024 | 64 | 128 | 64 |
| Case 2 | 128 | 1 | 4096 | 64 | 128 | 64 |

评分程序使用的名义浮点计算量为：

```text
FLOPs = 2 × total_context_len × next_n × num_heads × dim
```

按平均上下文长度估算，每次算子调用的计算量为：

| 测试用例 | 计算量 |
|---|---:|
| Case 1 | 约 1.074 GFLOPs |
| Case 2 | 约 8.590 GFLOPs |

以一个完整 KV Page 为单位，计算形状为：

```text
Q[64,128] × Kᵀ[128,64] -> Score[64,64]
```

单个 Page 的计算量为：

```text
2 × 64 × 64 × 128 = 1,048,576 FLOPs
```

### 1.3 计算特点

该算子具有以下特点：

1. **计算规模固定且规整。** Head 数、Dim 和 Page Size 分别固定为 64、128 和 64，适合手工展开和固定尺寸向量内核。
2. **KV Cache 是分页布局。** 逻辑上连续的 Token 在物理内存中按 Page 离散存放，无法直接把整个上下文视为连续矩阵。
3. **BF16 输入、FP32 累加。** 如果先把 KV 全部转换为 FP32，会增加临时内存、数据流量和转换开销；硬件 BF16 点积指令可以直接消费 BF16 数据并累加到 FP32。
4. **同一个 KV Page 会被 64 个 Head 重复使用。** Page 大小为 `64×128×2 = 16 KiB`，可以放入每核 32 KiB L1 Data Cache，适合以 Page 为单位组织计算。
5. **后处理可融合。** ReLU、Head 权重和 Head 归约不需要生成完整的中间 Score 矩阵，可以在点积完成后立即融合。
6. **计算量主要集中在 128 维 BF16 点积。** 输出初始化、分页索引和 Mask 是次要开销；点积内核的指令效率决定最终性能。

## 2. 计算环境分析

### 2.1 CPU 与存储层次

| 项目 | 配置 |
|---|---|
| 指令集架构 | AArch64 |
| CPU 厂商 | HiSilicon |
| CPU Part | `0xd22` |
| CPU 数量 | 608 核 |
| Socket | 2 × 304 核 |
| SMT | 无，1 Thread/Core |
| 最高主频 | 2.0 GHz |
| NUMA | 32 个节点，每节点约 38 核 |
| L1 Data Cache | 32 KiB/Core |
| L1 Instruction Cache | 32 KiB/Core |
| L2 Cache | 768 KiB/Core |
| Cache Line | 64 Bytes |

测试通常使用 32 个 OpenMP 线程，并将线程绑定到相邻 CPU Core。数据分配器在 ARM 平台通过 NUMA 接口将内存绑定到目标节点，以减少跨 NUMA 节点访存。

### 2.2 编译环境

| 项目 | 配置 |
|---|---|
| 操作系统工具链 | openEuler 3.0.3 |
| 编译器 | GCC 12.3.1 |
| OpenMP | 支持 |
| NUMA Library | 支持 |
| ARM ACLE Headers | `arm_sve.h`、`arm_bf16.h`、`arm_sme.h` 均可用 |

比赛版本使用的主要编译参数为：

```bash
g++ -O3 -march=armv9-a+sme+sve2 -fopenmp -lnuma main.cpp -o main
```

### 2.3 SVE 配置与实测结果

通过运行时接口、Intrinsic 测试和反汇编确认：

| 项目 | 实测结果 |
|---|---:|
| SVE Vector Length | 512 bit |
| SVE Vector Length | 64 Bytes |
| BF16 Lanes/Vector | 32 |
| FP32 Lanes/Vector | 16 |
| SVE2 | 支持 |
| SVE BF16 `BFDOT` | 支持 |

GCC 特性宏的实测结果为：

```text
__ARM_FEATURE_SVE=1
__ARM_FEATURE_SVE_BITS=0
__ARM_FEATURE_SVE_BF16=undefined
__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=1
```

其中，`__ARM_FEATURE_SVE_BITS=0` 表示使用长度无关的 SVE 编程模型，不代表实际向量长度为 0。运行时通过 `svcntb()` 和 `PR_SVE_GET_VL` 测得实际长度均为 64 Bytes。

尽管厂商 GCC 没有定义 `__ARM_FEATURE_SVE_BF16`，但 `svbfdot_f32` Intrinsic 可以正常编译。反汇编确认生成真实硬件指令：

```asm
bfdot z0.s, z1.h, z2.h
```

128 维 BF16 点积可以分成四个 512-bit 向量块：

```text
128 / 32 = 4
```

因此每个点积只需要四次 Query/KV 向量加载和四条 `BFDOT`，最后对 16 个 FP32 Lane 做一次横向归约。

### 2.4 SME 配置与实测结果

虽然当前优化版本主要使用 SVE，调试阶段同时验证了 SME，为后续矩阵化优化提供依据：

| 项目 | 实测结果 |
|---|---:|
| SME | 支持 |
| SME Streaming Vector Length | 512 bit |
| ZA Storage | 支持 |
| BF16→FP32 `BFMOPA` | 支持 |
| `arm_sme.h` | 可用 |

反汇编确认生成并成功运行：

```asm
smstart
zero   {za}
bfmopa za0.s, p0/m, p0/m, z0.h, z1.h
smstop
```

在 512-bit SME SVL 下，`ZA0.S` 对应一个 `16×16` FP32 Tile。一条 `BFMOPA` 完成：

```text
ZA[16,16] += A[16,2] × B[2,16]
```

因此 128 维 K 方向需要 64 条 `BFMOPA` 完成一个 `16×16` Tile。题目的 `64 Heads × 64 Tokens` Page 可以精确拆成 `4×4` 个 Tile，无尺寸尾部。

## 3. 优化历程

本次优化经历了三个主要版本：Baseline、Page-by-Page 框架版本和 SVE BF16 向量化版本。

### 3.1 Baseline：参考实现

Baseline 直接调用 `ref_bf16_paged_mqa_logits<float>`。其主要流程为：

1. 将整个有效 KV 上下文从 BF16 转换并展开为连续 FP32 `k_buffer`；
2. 为每个 Query Head 创建 FP32 `q_vec`；
3. 为每个 Head 创建上下文长度大小的 `scores`；
4. 逐 Head、逐 Token、逐 Dim 执行 FP32 点积；
5. 单独遍历 `scores` 执行 ReLU 和权重乘法；
6. 再次遍历 `scores`，沿 Head 维累加到输出。

Baseline 的优点是结构直观，且点积热循环已经是连续 FP32 数据，编译器可以进行一定程度的自动向量化。其主要问题包括：

- 完整上下文 KV 被扩展为 FP32，内存容量加倍；
- 存在上下文长度级别的临时缓冲区；
- 每个 Head 动态分配 `q_vec` 和 `scores`；
- Score 被初始化、计算、激活和归约多次遍历；
- 没有显式使用处理器的 BF16 点积能力。

实测性能：

| 测试用例 | Baseline |
|---|---:|
| Case 1 | 38.219666 GFLOPS |
| Case 2 | 42.419292 GFLOPS |

### 3.2 V1：Page-by-Page 计算框架

第一阶段将数据流改为以 KV Page 为单位处理。其循环结构为：

```text
batch
  query
    logical KV page
      head
        token
          dim
```

每个逻辑 Page 通过 `block_tables` 找到物理 Page，然后使用固定大小的：

```cpp
float page_output[64];
```

保存当前 Page 的 Head 归约结果。每个点积完成后立即融合：

```text
page_output[token] += weight[head] × ReLU(dot)
```

该版本实现了以下结构性优化：

- 不再构造完整上下文的 FP32 `k_buffer`；
- 不再为每个 Head 动态分配 `q_vec` 和 `scores`；
- KV Page 保持 BF16 存储，单页仅 16 KiB，可驻留 L1 Cache；
- 融合 ReLU、Weight 和 Head Reduction；
- 将大规模临时缓冲区缩小为固定 64 个 FP32；
- 为后续 SVE/SME 内核建立清晰的数据边界。

但是，V1 尚未接入 BF16 硬件指令，仍在最内层循环执行标量转换：

```cpp
dot += to_float(q[d]) * to_float(k[d]);
```

这使它处于一个性能较差的中间状态。Baseline 中 Q 只按 Head 转换一次，K 只按上下文转换一次；V1 却在每个 Head-Token 点积内反复将 Q 和 K 从 BF16 转换为 FP32。

以平均长度估算，Q 元素的重复转换次数达到：

- Case 1：每个 Q 元素约重复转换 1024 次；
- Case 2：每个 Q 元素约重复转换 4096 次。

同时，点积内环包含 BF16 转换操作，不再是简单的连续 FP32 FMA，限制了编译器自动向量化。因此，尽管 Page 化降低了临时内存和 KV 工作集，额外的标量转换成本仍超过了缓存收益。

实测性能：

| 测试用例 | Page-by-Page V1 | 相对 Baseline |
|---|---:|---:|
| Case 1 | 26.374455 GFLOPS | 69.01% |
| Case 2 | 28.682253 GFLOPS | 67.62% |

该版本性能下降约 31%～32%，但验证了新的分页框架、因果 Mask 和融合归约逻辑完全正确，为后续向量化排除了算法层面的风险。

### 3.3 V2：SVE BF16 `BFDOT` 向量化

第二阶段保持 V1 的 Page-by-Page 数据框架，仅将 ARM 点积内核替换为 SVE BF16 指令。x86 路径仍保留软件 BF16 转换，用于本地透明验证。

对于固定的 128 维点积，SVE 路径将 Query 分为四段：

```cpp
q0 = load(q + 0);
q1 = load(q + 32);
q2 = load(q + 64);
q3 = load(q + 96);
```

每段包含 32 个 BF16。当前 Head 的四个 Query 向量只加载一次，并在当前 Page 的所有 Token 之间复用。每个 Token 的点积执行：

```cpp
acc = svbfdot_f32(acc, q0, load(k + 0));
acc = svbfdot_f32(acc, q1, load(k + 32));
acc = svbfdot_f32(acc, q2, load(k + 64));
acc = svbfdot_f32(acc, q3, load(k + 96));
dot = svaddv_f32(acc);
```

相比 V1，该版本具有以下优势：

1. BF16 数据直接由硬件指令消费，不再显式转换为 FP32；
2. 一条 `BFDOT` 同时处理 32 个 BF16 乘法，并累加到 16 个 FP32 Lane；
3. 每个 128 维点积只需要四条核心点积指令；
4. Query 向量在一个 Page 内跨 64 个 Token 复用；
5. 16 KiB BF16 KV Page 可以有效利用 L1 Data Cache；
6. 保留 V1 中已完成的 ReLU、Weight、Head Reduction 融合；
7. 不再创建完整 FP32 KV 副本，减少内存带宽和 Cache 污染。

实测性能：

| 测试用例 | SVE V2 | 相对 Baseline | 相对 Page V1 |
|---|---:|---:|---:|
| Case 1 | 641.099136 GFLOPS | 16.774× | 24.307× |
| Case 2 | 729.706728 GFLOPS | 17.202× | 25.441× |

两个用例相对 Baseline 均达到约 17 倍加速。其中 Case 2 的上下文更长，Page 化后 KV 工作集和 BF16 带宽优势更加明显，最终性能也更高。

### 3.4 V3：SME Page 矩阵化

SVE V2 每次只计算一个 Head 与一个 Token 的点积，每个点积仍需要一次 FP32 横向归约。V3 将一个 KV Page 的计算重新表达为：

```text
S[64,64] = Q[64,128] × Kᵀ[128,64]
O[1,64]  = W[1,64] × ReLU(S[64,64])
```

第一阶段使用 SME `BFMOPA`，第二阶段使用 SVE 完成 ReLU、权重乘法和 Head 归约。第二阶段只有一行输出，且 ReLU 位于两次矩阵运算之间，因此 SVE 融合归约比再次调用 SME 更合适。

#### 3.4.1 Tile 划分与四 ZA 并行

512-bit SME SVL 下，一个 FP32 ZA Tile 为 `16×16`。对于一个 16-token Block，内核同时使用四个 ZA Tile：

```text
ZA0.S: heads  0..15 × 16 tokens
ZA1.S: heads 16..31 × 16 tokens
ZA2.S: heads 32..47 × 16 tokens
ZA3.S: heads 48..63 × 16 tokens
```

每轮 K-pair 只加载一次 K 向量，并供四条 BFMOPA 复用：

```asm
bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
bfmopa za1.s, p0/m, p0/m, z1.h, z4.h
bfmopa za2.s, p0/m, p0/m, z2.h, z4.h
bfmopa za3.s, p0/m, p0/m, z3.h, z4.h
```

一条 BFMOPA 推进两个 BF16 Dim，因此 64 轮完成一个 `64×16×128` Block。完整 64-token Page 包含四个 Token Block，共执行 1024 条 BFMOPA，对应 1,048,576 FLOPs。

#### 3.4.2 Packing 与 SVE 后处理

BFMOPA 输入围绕相邻 K-pair 排列：

```text
packed_q[k_pair][head_block][head][2]
packed_k[token_block][k_pair][token][2]
```

Q 每个 Query 打包一次并在全部 KV Page 间复用；K 每 Page 打包一次，Case 1 的两个 Query 共享该 K Packing。第一版将 ZA Tile 写入线程私有 `Score[64,64]`，再由 SVE 对 16 个 Token 并行执行：

```text
output += weight[head] × ReLU(score[head])
```

线程私有 Scratch 最大为 64 KiB，包括两个 packed Q、一个 packed K 和一个 FP32 Page Score。

#### 3.4.3 实测结果

| 测试用例 | SVE V2 | SME V3 | 相对 Baseline | 平均耗时 |
|---|---:|---:|---:|---:|
| Case 1 | 641.099136 GFLOPS | 5860.18 GFLOPS | 153.329× | 182.86 μs |
| Case 2 | 729.706728 GFLOPS | 4322.48 GFLOPS | 101.899× | 1997.24 μs |

两个用例的 `cos_diff` 均约为 `2e-14`。矩阵化一次生成 256 个 Score，消除了逐点 SVE 版本的大量横向归约，带来又一次数量级提升。Case 1 的 K Packing 可供两个 Query 复用，而 Case 2 每次 Packing 只服务一个 Query，因此后者更易受数据重排成本影响。

### 3.5 V4：SVE Gather 加速 K Packing

V3 对每个 K-pair 使用 16 次标量复制，从 16 个 Token Row 收集两个相邻 BF16。每个 Page 共执行：

```text
4 token blocks × 64 k-pairs × 16 copies = 4096 次复制
```

两个相邻 BF16 恰好占 32 bit，相邻 Token Row 的距离固定为 `128×2=256 Bytes`。V4 在进入 Streaming Mode 前使用普通 512-bit SVE Gather，偏移向量为：

```text
0, 256, 512, ..., 15×256 Bytes
```

一条 `svld1_gather_u32offset_u32` 收集 16 个 Token 的 BF16 Pair，再用一次连续 `svst1_u32` 写成 BFMOPA 需要的 64-byte Packed Vector。每 Page 的 Packing 主体缩减为：

```text
256 次 Gather + 256 次连续 Store
```

该过程只搬运原始 32-bit 数据，不进行数值转换，也不改变 BFMOPA 输入和累加顺序。实测结果如下：

| 测试用例 | SME V3 | SME V4 | V4/Baseline | 平均耗时 |
|---|---:|---:|---:|---:|
| Case 1 | 5860.18 GFLOPS | 7798.54 GFLOPS | 204.045× | 137.41 μs |
| Case 2 | 4322.48 GFLOPS | 6600.30 GFLOPS | 155.597× | 1307.98 μs |

V4 相对 V3 分别提升 33.08% 和 52.70%。Case 2 的单个 K Packing 只服务一个 Query，V3 中 Packing 占比更高，因此将标量重排替换为 SVE Gather 后收益更明显。Case 1 的同一份 K Packing 可以供两个 Query 复用，Packing 原本已经被摊薄，因而相对增益略低。

两个用例的 `cos_diff` 均满足 `< 5e-6`，说明 Gather 仅改变了数据搬运方式，没有改变 BF16 输入、SME 点积、ReLU 和加权归约的数值语义。

### 3.6 V5：消除 Page Scores 中间缓冲（尝试，未采纳）

V4 的数据流包含一次完整的 ZA → memory → SVE 中转：

```text
BFMOPA → ZA[16×16] → st1w → page_scores[64×64] (16 KB) → ld1w → SVE 后处理
```

V5 尝试将后处理（ReLU、Head Weight、Head Reduction）熔入 SME streaming mode，消除 page_scores 中转。核心思路：每 tile 计算完 BFMOPA 后，不写回 page_scores，而是逐行从 ZA 读出、在 SVE 寄存器内完成 ReLU + 加权归约，直接累加到输出的 `[16]` 向量。

实现方案：

```text
BFMOPA (64 kp) → ZA[16×16]
  → 逐行 st1w + ld1w（L1 中转 64 bytes）
  → fmax ReLU
  → ldr + dup 加载标量 weight
  → fmla 累加到 result[16]
  → 存 output
```

实测结果：

| 测试用例 | V4 (GFLOPS) | V5 (GFLOPS) | 降幅 |
|---|---:|---:|---:|
| Case 1 | 7798.54 | 4510.99 | -42.2% |
| Case 2 | 6600.30 | 4288.69 | -35.0% |

精度满足 `< 5e-6`。性能下降的原因分析：

1. **逐行 st1w + ld1w 增加了 L1 读写流量。** 每个 tile 16 行 × 4 tile = 64 次 64-byte 的 store+load（4 KB 额外 L1 流量），V4 一次 st1w 写 16 KB 可以充分发挥写合并带宽。
2. **逐 head 加载权重的 IPC 瓶颈。** V4 用 `ld1w` 一次加载 16 个 weight，展开的 64 次 `svmla_n_f32` 循环编译器可以有效调度；V5 的 `ldr + dup + fmla` 每次只有一条 fmla 做计算，加载权重需要额外的地址计算和数据搬运指令，占用指令发射带宽。
3. **SME 外区域指令开销。** ZA 行存储（`st1w`）和 SVE 加载（`ld1w`）在 streaming mode 内各占不同的执行流水线，但 64 次的小循环体无法充分利用发射队列，导致流水线气泡。

结论：消除 page_scores 在方向上是正确的，但当前实现增加的 L1 流量和逐标量 weight 加载开销超过了消除大缓冲带来的收益。保留 page_scores 的 V4 方案在现有硬件上更优。

