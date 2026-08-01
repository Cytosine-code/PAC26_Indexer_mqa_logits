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
packed_q[4 k_pair][4 head_block][4 head][2 dim]
packed_k[4 token_block][64 k_pair][16 token][2 dim]
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

**另外**

> 我们再次尝试了ZA -> MOVA -> Z的page scores消除方案，尽管确实是比当前的ZA -> 栈 -> Z方案好一点，但依旧带来严重负优化，故放弃此优化路线。说明，在 LX2 上，16 KiB `page_scores` 是高效的 L1 解耦缓冲。直接使用 MOVA 在 Streaming Mode 中融合后处理，减少了内存流量，却增加了 ZA Slice 读取延迟、Streaming SVE 成本和串行依赖，最终性能下降。



### 3.7 V6：SVE Gather 加速 Q Packing

V4 已使用 SVE Gather 优化 K Packing，但 Q Packing 仍按 Head 执行标量 BF16 Pair 复制。Q 与 K 的原始 Row Stride 相同，均为：

```text
128 BF16 × 2 Bytes = 256 Bytes
```

因此 Q Packing 可以复用 K Packing 的方法。对每个 `(k_pair, head_block)`，使用一条 16-lane 32-bit Gather，从 16 个 Head Row 收集同一对相邻 BF16，再连续写入 64-byte Packed Vector：

```cpp
const svuint32_t pairs = svld1_gather_u32offset_u32(
    pack_pg, reinterpret_cast<const uint32_t *>(src), pack_offsets);
svst1_u32(pack_pg, reinterpret_cast<uint32_t *>(dst), pairs);
```

每个 Query 的 Q Packing 主体由：

```text
64 k-pairs × 4 head blocks × 16 scalar copies
= 4096 次 Pair Copy
```

缩减为：

```text
256 次 SVE Gather + 256 次连续 Store
```

实测结果：

| 测试用例 | V4 | V6 | V6/V4 | 平均耗时 |
|---|---:|---:|---:|---:|
| Case 1 | 7798.54 GFLOPS | 8024.85 GFLOPS | 1.0290× | 133.54 μs |
| Case 2 | 6600.30 GFLOPS | 6794.49 GFLOPS | 1.0294× | 1270.59 μs |

两个用例分别提升 2.90% 和 2.94%，精度检查通过。虽然 Case 1 的 GFLOPS 绝对增量更大，但两者的相对增益和耗时降幅几乎一致，说明 Q Packing 在两个用例总耗时中的占比均约为 3%。

### 3.8 V7：收缩输出初始化并预取下一物理 Page

V7 包含两项互不改变主计算的外围优化。

第一项是只初始化因果有效区之后的 Mask 尾部。原实现先把完整的 8192 个输出元素写成 `-INFINITY`，随后又覆盖有效区；新实现仅执行：

```cpp
std::fill(out + valid_length, out + max_model_len, -INFINITY);
```

若 `block_tables` 中出现负物理 Block，则单独将该 Page 对应的有效区写成 `-INFINITY`，从而保持参考实现语义。

第二项是针对随机 Page 映射预取下一物理 KV Page。每个 Page 为 16 KiB，在当前 Page Packing 前，以 2 KiB 间隔发出 8 个 locality=2 的预取提示，使下一 Page 向 L2 靠近，同时避免强行占据当前 L1。

该版本短时测试结果存在明显波动：

| 测试轮次 | Case 1 | Case 2 |
|---|---:|---:|
| 第 1 次 | 7.87 TFLOPS | 6.89 TFLOPS |
| 第 2 次 | 8.07 TFLOPS | 6.92 TFLOPS |

Case 1 在一次测试中低于 V6、另一次略高于 V6；Case 2 两次均略高于 V6，但幅度仅约 1%～2%。由于 Case 1 单次平均耗时仅约 134 μs，数微秒波动即可造成明显的 GFLOPS 变化。该项优化未观察到稳定负优化，且减少了理论冗余工作，因此予以保留，但不将其收益计入稳定版本加速结论。

### 3.9 V8：BFMOPA 双缓冲软件流水

原 SME 主循环每个 K-pair 依次执行 5 条 Load 和 4 条 BFMOPA，随后才开始加载下一组。V8 使用两套 Z 寄存器实现奇偶 K-pair 双缓冲：

```text
偶数组：z0-z4
奇数组：z5-z9
```

执行顺序调整为：

```text
预加载 pair 0
加载 pair 1 -> 计算 pair 0
加载 pair 2 -> 计算 pair 1
加载 pair 3 -> 计算 pair 2
...
```

目标是让下一组 Q/K Load 与当前组 BFMOPA 在处理器流水线上重叠。循环采用 31 次双迭代和最后一对尾处理，不会越界读取。ZA 的实际累加顺序仍严格保持 `pair 0,1,...,63`，因此没有改变浮点结合顺序。

实测结果：

| 测试用例 | V6 稳定基线 | V8 流水版 | 相对 V6 |
|---|---:|---:|---:|
| Case 1 | 8024.85 GFLOPS | 8093.71 GFLOPS | 1.0086× |
| Case 2 | 6794.49 GFLOPS | 6805.02 GFLOPS | 1.0015× |

精度检查通过。Case 1 提升约 0.86%，Case 2 提升约 0.15%，收益较小。这表明 LX2 对原始 Load/BFMOPA 循环已经具备较好的乱序调度能力，或者 SME 矩阵执行单元的 ZA 累加吞吐比 Load 延迟更接近当前瓶颈。双缓冲增加了寄存器使用和循环体尺寸，但没有产生明显负优化。

但是，汇编代码的优化很困难且该收益很可能被运行波动覆盖。为了降低代码复杂度、便于后续优化，V8 的双缓冲实现已回退，后续版本继续建立在 V7 主线上。

### 3.10 V9：使用 SME ZA 转置优化 K Packing

#### 3.10.1 性能瓶颈

阶段计时显示，V7 的主要剩余瓶颈已经从矩阵计算逐步转移到 K Packing：

| 测试用例 | Q Pack | K Pack | SME | Postprocess | Mask Init |
|---|---:|---:|---:|---:|---:|
| Case 1 | 4.41% | 35.56% | 43.56% | 12.27% | 4.20% |
| Case 2 | 0.98% | 54.47% | 35.00% | 9.11% | 0.44% |

V4 的 SVE Gather 已经比标量复制快很多，但每个 16 KiB Page 仍需要：

```text
4 token blocks × 64 k-pairs
= 256 次 16-lane Gather + 256 次连续 Store
```

Gather 的 16 个输入地址间隔 256 Bytes，加载延迟和地址生成成本较高。原始 K Page 实际上可以视为：

```text
K[64 tokens][64 BF16-pairs]
```

目标 Packed K 则是在每个 16-token Block 内进行 `16×16 uint32` 转置。因此可以把 ZA 同时作为矩阵寄存器和数据转置器使用。

#### 3.10.2 ZA 转置方案

每个 K Page 被拆成 16 个 `16×16 uint32` Tile。对每个 Tile：

1. 使用 16 条连续 `ld1w {za0h.s[...]}`，将源矩阵的 16 行载入 ZA 横向 Slice；
2. 使用 16 条连续 `st1w {za0v.s[...]}`，从 ZA 纵向 Slice 写出目标矩阵的 16 列；
3. 整个过程只搬运 BF16 Pair 的原始 32-bit 位模式，不执行数值转换。

完整 Page 的主体指令数量变为：

```text
256 次连续 512-bit Load + 256 次连续 512-bit Store
```

向量搬运条数与 Gather 版本相同，但输入端由 256 次离散 Gather 变为 256 次连续 ZA Slice Load，降低了地址生成和离散访存成本。

独立探针在 LX2 上完成了逐元素布局校验：

```text
SME SVL: 512 bits
PASS: SME K transpose matches gather packing
hot-page gather: 2967.59 ns/page
hot-page SME:     619.60 ns/page
speedup:          4.790x
```

探针验证了以下真实指令：

```asm
smstart
ld1w {za0h.s[w12, 0]}, p0/z, [x5, xzr, lsl #2]
st1w {za0v.s[w12, 0]}, p0, [x6, xzr, lsl #2]
smstop
```

接入算子时还需要遵守 AAPCS64：`smstart/smstop` 会影响向量寄存器状态，因此汇编函数在进入 Streaming Mode 前保存 `d8-d15`，退出后恢复。缺少该保护时，编译器保存在这些寄存器中的阶段计时累加器会被破坏，表现为 `denom == 0`；修复后阶段统计恢复正常。

#### 3.10.3 完整算子结果

接入后，阶段计时中 Case 1 的 K Packing 约降低 `0.01 ms`，Case 2 约降低 `0.10 ms`。整算子的六轮测试结果如下：

| 轮次 | Case 1 | Case 2 |
|---:|---:|---:|
| 1 | 8.420979 TFLOPS | 7.568968 TFLOPS |
| 2 | 8.535716 TFLOPS | 7.546321 TFLOPS |
| 3 | 8.468822 TFLOPS | 7.552872 TFLOPS |
| 4 | 8.457002 TFLOPS | 7.557757 TFLOPS |
| 5 | 8.567334 TFLOPS | 5.991723 TFLOPS |
| 6 | 5.300648 TFLOPS | 7.555746 TFLOPS |

稳定样本的中位数与此前最快稳定版本对比如下：

| 测试用例 | 此前最快版本 | V9 中位数 | 提升 |
|---|---:|---:|---:|
| Case 1 | 8.09 TFLOPS | 8.463 TFLOPS | 4.6% |
| Case 2 | 6.80 TFLOPS | 7.554 TFLOPS | 11.1% |

V9 的完整算子收益显著小于热页探针的 4.79 倍。这是因为探针重复处理常驻缓存的同一 Page，主要测量数据重排指令；完整算子还包含随机物理 Page 的冷缓存访问、多核共享缓存及内存带宽竞争，以及每 Page 的 Streaming Mode 切换。ZA 转置优化消除了大部分 Gather 指令成本，但无法消除 16 KiB 输入读取和 16 KiB Packed K 写入。

#### 3.10.4 性能异常与风险

六轮测试中出现两个孤立异常：Case 2 第 5 轮下降至 5.99 TFLOPS，Case 1 第 6 轮下降至 5.30 TFLOPS。异常发生在不同轮次和不同测试用例，同轮的另一个测试用例仍保持正常性能，因此目前没有证据表明它由固定输入触发的算法分支或确定性的 K Packing 路径造成。

但由于此前版本未观察到同等幅度的下降，V9 暂时保留以下待验证风险：

1. 频繁进入 SME Streaming Mode 是否放大了线程抢占或迁移的代价；
2. OpenMP 线程是否始终固定在同一 NUMA 节点和同一组核心；
3. 异常轮次的耗时是否集中在 K Packing、SME，还是整个算子所有阶段同步增加；
4. 多核同时执行 ZA 转置时是否触发共享缓存或内存带宽的瞬时拥塞。

后续应在固定 `OMP_NUM_THREADS`、`OMP_PROC_BIND`、`OMP_PLACES` 和 NUMA 绑定的条件下重复测试，并保留异常轮次的原始阶段计时。当前依据中位数判断，V9 相对稳定版本具有明确收益，予以保留。

#### 3.10.5 删除 Packed K 的片上直算探索（未采纳）

V9 之后曾尝试不落地 `packed_k`，直接把原始 K Page 分块装入 ZA，再通过 ZA Slice 与 Z 寄存器之间的数据移动完成 BFMOPA。独立探针首先验证了结果与物化 Packed K 完全一致，但逐 Tile 版本明显变慢：

| 框架 | 耗时 | 吞吐 |
|---|---:|---:|
| 物化 Packed K | 647.40 ns/tile | 404.92 GFLOPS |
| Direct K | 1012.52 ns/tile | 258.90 GFLOPS |

进一步把多个 Tile 合并到一次 Streaming Mode 中，并批量复用寄存器后，Direct K 才与物化框架基本持平：

| 框架 | 耗时 | 吞吐 |
|---|---:|---:|
| 物化 Packed K | 648.54 ns/tile | 404.21 GFLOPS |
| Batched Direct K | 642.57 ns/tile | 407.96 GFLOPS |

即使在热数据探针中，Direct K 的加速也仅为 `1.009x`。它没有提供足够余量抵消更复杂的寄存器调度、`next_n` 复用退化和边界处理成本，因此没有接入正式算子。该实验说明：在 LX2 上，`packed_k` 虽然占用显著时间，但它同时把后续 BFMOPA 所需的数据布局预先连续化；简单删除中间缓冲并不等价于删除这部分工作。

### 3.11 V10：Page 级并行调度框架

#### 3.11.1 原有调度的负载不均衡

V9 及以前使用 OpenMP 按 Batch 静态并行：

```cpp
#pragma omp parallel for schedule(static)
for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    // 一个线程完成该 batch 的全部 pages
}
```

设第 \(b\) 个 Batch 的上下文长度为 \(L_b\)，每个 Page 包含 64 个 Token，则它的 Page 数为
$$
N_b = \left\lceil \frac{L_b}{64} \right\rceil 
$$
若线程 \(t\) 获得的 Batch 集合为 \(\mathcal{B}_t\)，忽略边界 Page 的微小差异，其计算量近似为

$$
W_t \propto \sum_{b \in \mathcal{B}_t} N_b \cdot C_{\mathrm{page}}(\text{next\_n})
$$
由于各 Batch 的 `context_len` 存在波动，而 OpenMP 只保证每个线程获得相近数量的 Batch，并不保证 \(\sum N_b\) 相近，因此总耗时由 Page 最多的线程决定：

$$
T_{\mathrm{parallel}} \approx \max_t W_t
$$
Case 1 的 `batch_size=32`，最多只能自然使用 32 个线程；Case 2 虽然有 128 个 Batch，但每个线程只获得少数几个完整 Batch，随机长度差异仍会形成长尾。这使 38 核不能被充分利用。

#### 3.11.2 Page 前缀和与等量切分

V10 把并行任务的基本单位从 Batch 改为 Page。先构造 Page 前缀和：$P_0=0, \qquad P_{b+1}=P_b+N_b,$

其中 \(P_B\) 是全部 Batch 的总 Page 数。对 \(T\) 个 OpenMP 线程，线程 \(t\) 处理全局 Page 区间

$$
s_t=\left\lfloor\frac{P_Bt}{T}\right\rfloor, \qquad
e_t=\left\lfloor\frac{P_B(t+1)}{T}\right\rfloor
$$
因此任意两个线程的 Page 数量最多相差 1：

$$
\max_t(e_t-s_t)-\min_t(e_t-s_t)\le 1
$$
线程只在区间起点使用一次 `upper_bound`，由 \(s_t\) 定位起始 Batch 和 Batch 内 Page 编号，之后沿 Batch-major 顺序线性推进。该设计没有 OpenMP Task、原子计数器或动态任务队列，避免在每个 Page 上引入调度同步；同时连续 Page 区间仍保留较好的 Q 与 Page Table 局部性。

按平均长度估算，两组用例的任务粒度为：

| 测试用例 | 总 Page 数 | 38 线程下每线程 Page 数 |
|---|---:|---:|
| Case 1 | $32\times1024/64=512$ | 13--14 |
| Case 2 | $128\times4096/64=8192$ | 215--216 |

这使 Case 1 可以突破原先 32 个 Batch 带来的并行度上限，也基本消除了 Case 2 的 Batch 长度长尾。

#### 3.11.3 两阶段单并行域

Page 级并行要求任意线程都能读取任意 Batch 的 Packed Q。V10 因此把算子组织为同一个 OpenMP Parallel Region 内的两个阶段：

1. **Q Packing 阶段**：所有线程用 `omp for` 并行生成共享的 `packed_q[batch][next_n]` 和 Mask；
2. **Page 阶段**：利用 `omp for` 末尾的隐式 Barrier，保证 Packed Q 全部可见，然后每个线程执行自己的连续 Page 区间。

共享 Packed Q 的容量为
$$
B\times N\times64\times128\times2\ \text{Bytes}
$$
即 Case 1 约 1 MiB、Case 2 约 2 MiB。该空间通过持久化 Workspace 复用，避免每次算子调用重复分配。`packed_k` 和 `page_scores` 仍为线程私有；对每个 Page，线程在同一核心上连续完成

```text
K Packing -> 所有 next_n 的 SME 点积 -> ReLU/Head Reduce -> 写回
```

因此 V10 只改变任务所有权和同步位置，不改变 V9 已验证的 K Packing、BFMOPA 及后处理数值路径。每个 Page 的输出区间互不重叠，不需要写入锁。

#### 3.11.4 NUMA 绑定问题与修正

初次使用 38 线程时，性能反而下降至约 6--7 TFLOPS。原因不是 Page 调度算法，而是运行脚本使用了：

```bash
taskset -c 1-38
```

CPU 编号区间两端均包含在内；当单 NUMA 节点的计算核心为 `0-37` 时，`1-38` 会漏掉 CPU 0，并错误包含相邻 NUMA 节点的 CPU 38。32 线程使用 `1-32` 时恰好没有越界，因而问题只在扩展到 38 线程后暴露。

V10 的共享 Workspace 和输入内存具有明确 NUMA 归属。一旦某个工作线程跨节点，它既可能产生远程内存访问，也可能成为并行区最后完成的线程；由于壁钟时间取决于最慢线程，单个跨 NUMA 核就足以显著拉低整算子性能。将 CPU 集合修正为严格的 `0-37`，并固定


```bash
OMP_NUM_THREADS=38
OMP_PROC_BIND=close
```

后，异常性能消失。该实验也证明，CPU/NUMA 亲和性错误确实能够造成与 V9 异常轮次同量级的性能暴跌；但 V9 当时是否使用了越界核心集合缺少记录，因此不能据此把 V9 的偶发波动直接归因于同一问题。

V10 的核心收益不是提高单个 SME Kernel 的峰值吞吐，而是让更多核心持续执行有效 Page 工作。尤其在 Case 2 中，性能再次高于 Case 1，说明原先由 Batch 粒度造成的负载不均衡已被大幅消除；当前瓶颈重新集中到每个 Page 内部的 K 搬运、BFMOPA 计算和固定同步开销。

### 3.12 V10x：V10 框架上的小粒度调优

V10 完成 Page 级任务均衡后，继续大幅修改调度框架的收益与风险已经不匹配。V10x 因此保留 V10 的共享 Packed Q、Page 前缀和与 38 核静态切分，只优化每个 Page 都会重复执行的固定成本。V10x 初始包含三个彼此独立的小改动，随后又加入满 Page `2×2 BFMOPA` 寄存器分块。

#### 3.12.1 V10x.1：融合 K Packing 与第一个有效 Q

V10 中，每个有效 Page 先调用 ZA 转置函数生成 Packed K，然后调用 BFMOPA 函数计算 Page Scores。分离路径需要经历两组函数序言、寄存器保护和 Streaming Mode 切换：

```text
保存 d8-d15
smstart -> K Packing -> smstop
恢复 d8-d15

保存 d8-d15
smstart -> BFMOPA -> smstop
恢复 d8-d15
```

V10x.1 把 K Packing 主体与第一个有效 Q 的 BFMOPA 主体连接到同一个汇编入口：

```text
保存 d8-d15
smstart -> K Packing -> BFMOPA -> smstop
恢复 d8-d15
```

该改动消除了一次 `smstop -> smstart` 往返、一组 `d8-d15` 保存恢复和一次函数调用。Packed K 中间缓冲仍然保留，因此没有重复此前“删除 Page Scores”或 Direct K 方案的失败；当 `next_n=2` 时，第二个 Q 继续复用同一份 Packed K。

实现不能简单假设 `n=0` 永远有效。当上下文长度刚好跨过 Page 边界时，最后一个 Page 可能对 `n=0` 没有有效 Token、但对 `n=1` 有效。因此代码使用 `packed_k_ready` 状态，由第一个满足 `valid_tokens>0` 的 Q 触发融合入口，保证尾 Page 语义正确。

为了保留阶段 Profiler 中 K Packing 与 SME 的独立计时，`EN_TIMING` 构建继续使用分离路径；V10x.1 的真实性能必须通过关闭 Profiler 的正式构建测量。接入后，无 Profiler 性能稳定在约 `8.8--9.0 TFLOPS / 9.8--9.9 TFLOPS`，相对 V10 有明确提升。

#### 3.12.2 V10x.2：删除无效的下一 Page 软件预取

早期版本在处理当前 Page 前，对下一物理 Page 的 16 KiB K 数据抽样发出 8 条软件预取，每隔约 2 KiB 预取一条 64 Byte Cache Line。该方案只覆盖

$$
\frac{8\times64}{16\times1024}=3.125\%
$$

的下一 Page 数据，并且在 V10 的 38 核 Page 级并行框架下，每个线程已经沿连续的逻辑 Page 区间推进，原有预取的成本模型发生了变化。

曾尝试把预取加强到 32 条、间隔缩短为 512 Bytes，使覆盖率提高到 12.5%。结果平均性能下降到约 `8.5 / 9.4 TFLOPS`，Profiler 中 Case 2 的 K Packing 反而增加约 `0.04 ms`。原因是 38 个核心同时发出大量 `PRFM`，增加了前端、地址生成、缓存填充队列和内存并发请求压力，而抽样预取仍不能替代对完整 16 KiB Page 的读取。

随后进行了完全删除软件预取的 A/B 测试。10 轮测试的主体分布与 V10x.1 基本相同，并偶发达到 `9.00 / 10.00 TFLOPS`。这说明原来的 8 条预取没有可重复的正收益。V10x.2 最终删除该分支，以同等性能换取更短的 Page 热循环和更低的代码复杂度。

#### 3.12.3 V10x.3：四 Token Tile 联合后处理

每个 Page 的 SME 结果布局为：

$$
S\in\mathbb{R}^{64\times64},
$$

其中 64 行对应 Head，64 列对应 Token。后处理计算为

$$
O_t=\sum_{h=0}^{63}W_h\cdot\max(S_{h,t},0).
$$

SVE 每次处理 16 个 FP32 Token，因此完整 Page 被划分为 4 个 Token Tile。原循环以 Token Tile 为外层、Head 为内层：

```text
for tb in 0..3:
    result[tb] = 0
    for h in 0..63:
        result[tb] += relu(scores[h][tb]) * weight[h]
```

完整 Page 中，每个 Head 权重会被加载或广播 4 次，总计

$$
4\times64=256
$$

次。V10x.3 将满 Page 改为 Head 外循环，并同时维护 4 个 SVE 累加器：

```text
result0, result1, result2, result3 = 0
for h in 0..63:
    weight = broadcast(weights[h])
    result0 += relu(scores[h][0]) * weight
    result1 += relu(scores[h][1]) * weight
    result2 += relu(scores[h][2]) * weight
    result3 += relu(scores[h][3]) * weight
```

权重加载或广播次数由 256 次降为 64 次，减少 75%；四条独立 FMA 依赖链也提高了指令级并行度。Page Scores 的读取次数、FMA 数量以及每个 Token 沿 Head 维的累加顺序均保持不变，因此不会引入新的数值误差来源。

当 `token_tiles<4` 时，尾 Page 继续使用原循环；当第四个 Tile 只有部分有效 Token 时，使用 SVE Predicate 仅写回 `valid_tokens-48` 个结果。该优化接入后，实测性能达到：

| 测试用例 | V10x.3 实测性能 |
|---|---:|
| Case 1 | 9.2 TFLOPS |
| Case 2 | 10.01 TFLOPS |

Case 2 首次在正确性通过的正式算子中越过 10 TFLOPS。由于当前单次测试仍存在约 1% 量级波动，`10.01 TFLOPS` 应视为目前最好实测值；是否稳定站上 10 TFLOPS 仍需通过更多轮次的中位数与最差值确认。

#### 3.12.4 V10x.4：满 Page 2×2 BFMOPA

原 BFMOPA 内核使用 `4 Head Block × 1 Token Block` 分块。每个 K Pair 需要加载 4 个 Q 向量和 1 个 K 向量；完整 Page 的输入向量加载数为

$$
4\times64\times(4+1)=1280.
$$

新内核将四个 ZA Tile 映射为 `2 Head Block × 2 Token Block`：

```text
ZA0 = Head A × Token A
ZA1 = Head A × Token B
ZA2 = Head B × Token A
ZA3 = Head B × Token B
```

每个 K Pair 只需加载 2 个 Q 向量和 2 个 K 向量，完整 Page 的输入向量加载数变为

$$
4\times64\times(2+2)=1024,
$$

减少 20%。BFMOPA 数量仍为每 Page/Query 1024 条，每个输出 Tile 沿 K 维的累加顺序也保持不变。满 64 Token Page 使用新内核，尾 Page 继续使用已验证的 `4×1` 内核；第一个满 Page Q 仍与 K Packing 共用一次 Streaming Mode。

由于测试期间机器整体性能较先前下降，采用同一机器状态下重新测试的旧版本作为基线：

| 测试用例 | 2×2 前 | 2×2 后 | 同机提升 |
|---|---:|---:|---:|
| Case 1 | 8.523 TFLOPS | 8.929 TFLOPS | 4.76% |
| Case 2 | 9.444 TFLOPS | 10.178 TFLOPS | 7.77% |

该结果证明 2×2 分块减少的 Load 指令确实位于关键路径。Case 2 满 Page 比例更高、运行时间更长，因此获得了更充分的稳态收益。

#### 3.12.5 V10x 整体收益

以 V10 无 Profiler 最好成绩 `8.6 / 9.5 TFLOPS` 为历史参照，V10x 目前记录到的最好成绩为 `9.2 / 10.178 TFLOPS`，最好值之间的提升为

$$
\frac{9.2}{8.6}-1=6.98\%,
$$

$$
\frac{10.178}{9.5}-1=7.14\%.
$$

该比较是跨机器状态的最好值对比，并不等同于严格统计意义上的稳定加速。2×2 BFMOPA 的收益应以同机 `4.76% / 7.77%` 为准；其余改动分别消除了 Streaming Mode 往返、无效预取指令和重复权重广播，机制上相互独立。

### 3.13 使用 KUPL 调用层（未采纳）

#### 3.13.1 动机

Phase Timing 中 Case 1 的 Overhead 占比约 13%，主要来自每次算子调用重建 OpenMP region（约 9.4 μs）以及 Q Pack 与 Page 阶段之间的全局 barrier。KUPL 的 `kupl_parallel_for` 单次启动开销约 5.5 μs，预探针确认可以替代 OpenMP 调度层。因此尝试把调度层整体替换为 KUPL，计算主内核保持不变，专门验证调度差异。

#### 3.13.2 实现方案

计算路径（手写 SME 2×2 BFMOPA、ZA K Packing、SVE 后处理）与 V10x 完全一致，仅把调度层改为单个 `kupl_parallel_for(STATIC)`。为保证只用一次并行启动，把 Q Pack 移入 Page 工作线程：每个线程先打包自己 Page 区间所覆盖的 batch 的 Q，再处理这些 Page；区间边界的 batch 由相邻两个线程重复打包，写入值完全相同，且每个线程只读取自己打包过的行，因此无需任何全局 barrier。对应实现为 `kupl_mqa_logits.h`。

```text
kupl_parallel_for(STATIC) [0, total_pages)
        ↓
每个 worker：打包本区间覆盖的 Q → 逐 Page（K Pack → SME 2×2 → 后处理）
```

#### 3.13.3 实测结果

多轮同机测试的稳定结果为：

| 测试用例 | V10x 原版 | KUPL 调用层 | 相对变化 |
|---|---:|---:|---:|
| Case 1 | 9.2 TFLOPS | 8.8 TFLOPS | -4.3% |
| Case 2 | 10.4 TFLOPS | 10.3 TFLOPS | ~-1% |

Case 2 几乎不变，Case 1 反而小幅下降。

#### 3.13.4 未采纳原因

1. **Case 2 本就是带宽受限**：`next_n=1` 时算术强度固定为 64 FLOP/Byte，13.2 TFLOPS 的 K 带宽 roof 已约占 78%，调度层没有可优化空间，实测持平符合预期。
2. **Case 1 的 Q Pack 边界冗余**：Case 1 的 Q Pack 约占 Phase Timing 的 14%，合并式调度使边界 batch 被两个线程重复打包，打包量放大到约 1.5~2 倍，新增成本超过了省下的约 4 μs 全局 barrier 与启动开销。

因此调度层不是该算子的剩余瓶颈，该优化未接入正式版本。

#### 3.13.5 结论

KUPL 调用层替换为中性偏负收益。正式版本继续使用 V10x 框架（`indexer_mqa_logits.h`）；`kupl_mqa_logits.h` 作为参考保留，具有回退保护且结果正确。

## 4. 阶段性总结

### 4.1 性能演进

从参考实现到 V10x，优化路径经历了算法分块、SVE 向量化、SME 矩阵化、数据重排优化和并行调度重构：

| 版本 | 核心优化 | Case 1 | Case 2 |
|---|---|---:|---:|
| Baseline | 参考 Page/Head/Token/DIM 循环 | 0.0382 TFLOPS | 0.0424 TFLOPS |
| V2 | SVE BF16 BFDOT | 0.641 TFLOPS | 0.730 TFLOPS |
| V3 | SME BFMOPA Page 矩阵化 | 7.799 TFLOPS | 6.600 TFLOPS |
| V6 | Q Packing 优化 | 8.025 TFLOPS | 6.794 TFLOPS |
| V9 | SME ZA K Packing | 8.463 TFLOPS | 7.554 TFLOPS |
| V10 | Page 级负载均衡与 38 核调度 | 8.6 TFLOPS | 9.5 TFLOPS |
| V10x | 融合、删除预取、联合后处理、2×2 BFMOPA | **9.2 TFLOPS** | **10.178 TFLOPS** |

相对 Baseline，当前最好成绩的累计加速约为

$$
\frac{9.2}{0.038219666}=240.7\times,
$$

$$
\frac{10.178}{0.042419292}=239.9\times.
$$

### 4.2 主要经验

1. **先改变算法数据流，再优化指令。** Page-by-Page 框架和矩阵化将 K Page 从重复读取对象变为可复用计算块，是百倍加速的基础。
2. **数据布局与矩阵指令同等重要。** BFMOPA 提供高计算吞吐，但 Q/K Packing 一度占据 30%--50% 时间；ZA 转置和共享 Packed Q 决定了矩阵单元能否持续工作。
3. **负载均衡必须以真实工作量为单位。** 固定测试规模并不代表每个 Batch 的 Page 数相同。按 Page 前缀和切分后，Case 2 的 32 至 38 核扩展效率接近理想值。
4. **NUMA 与绑核是算法性能的一部分。** `taskset -c 1-38` 只错一个 CPU 编号，就能让性能下降数 TFLOPS；单 NUMA 节点必须严格绑定 `0-37`。
5. **微基准收益不能直接外推到完整算子。** 热页 ZA 转置达到 4.79 倍，但整算子还受到冷数据、缓存竞争和内存带宽限制；Direct K 热探针的 1.009 倍也不足以支撑复杂重构。
6. **中间缓冲不一定是浪费。** 两次删除 Page Scores、一次 Direct K 尝试都说明，中间布局可以换取规整访存、寄存器复用和更高矩阵吞吐。
7. **预取不是越多越好。** 加强 Page 预取使 K Packing 明显变慢；软件预取只有在覆盖、提前量和硬件请求容量匹配时才有价值。
8. **固定规模允许有针对性的寄存器分块。** 四 Token Tile 联合后处理利用了 `block_size=64` 与 SVE FP32 16 Lane 的固定关系，在不改变运算量的前提下减少权重广播并增加 ILP。

### 4.3 单 NUMA 真实 SME 峰值

公开资料中的整颗 304 核 CPU BF16 算力约为 240 TFLOPS，按 8 个 38 核簇平均只能得到约 30 TFLOPS。为了避免不同频率和测试口径造成误判，使用 38 核纯 BFMOPA 探针实测当前节点。

探针在每个线程的一次 `smstart` 内把四个操作数常驻 Z 寄存器，循环执行：

```asm
bfmopa za0.s, ..., z0.h, z4.h
bfmopa za1.s, ..., z0.h, z5.h
bfmopa za2.s, ..., z1.h, z4.h
bfmopa za3.s, ..., z1.h, z5.h
```

一条 512-bit BFMOPA 完成 1024 FLOPs，每轮四条，因此 38 线程、每线程 5,000,000 轮的总计算量为

$$
38\times5{,}000{,}000\times4\times1024
=778.24\text{ GFLOPs}.
$$

探针完成四个 ZA Tile 的逐元素校验，并确认 38 个线程分别绑定在 CPU `0-37`。七轮结果为：

| 指标 | 实测结果 |
|---|---:|
| 中位数 | 38.651 TFLOPS |
| 最好值 | 38.847 TFLOPS |

该结果与 2.0 GHz 下每核约每两周期发射一条 BFMOPA 的理论值吻合：

$$
38\times2\text{ GHz}\times
\frac{1024\text{ FLOPs}}{2\text{ cycles}}
=38.912\text{ TFLOPS}.
$$

实测中位数达到该值的 99.3%，因此本报告采用

$$
P_{\mathrm{SME}}=38.651\text{ TFLOPS}
$$

作为单 NUMA 的真实纯 BFMOPA 峰值。该峰值不包含操作数 Load、ZA Store、Packing、后处理和 Streaming Mode 切换，不能直接视为完整算子上限。

### 4.4 随机 K Page 有效带宽

带宽探针复现正式算子的关键条件：

1. 使用 512 MiB 数据集，远大于 38 核合计约 29 MiB L2；
2. CPU 固定为 `0-37`；
3. 按正式 Allocator 规则将数据绑定到 Memory Node 16；
4. 将 32768 个 16 KiB Page 随机排列，各线程读取互不重叠的 Page；
5. Page 内按 ZA K Packing 的真实次序访问。

Page 内地址为

$$
\mathrm{offset}
=tb\times4096+column\times64+row\times256,
$$

其中 `tb=0..3`、`column=0..3`、`row=0..15`。它恰好覆盖 16 KiB Page 的全部 256 条 64 Byte Cache Line，各访问一次。

| 访问方式 | 中位带宽 | 最好带宽 |
|---|---:|---:|
| 连续 Page + 连续 Cache Line | 385.03 GB/s | 391.21 GB/s |
| 随机 Page + K Packing 次序 | 206.25 GB/s | 207.88 GB/s |

随机 Page 结果只有连续读取的 53.6%，说明 Block Table 随机化和 Page 内跨行访问显著降低了内存系统效率。探针为了保证所有 Load 可观察，还包含 SVE XOR 归约和地址循环，因此 206.25 GB/s 应理解为该访问模式的有效读取吞吐，而不是纯 DRAM 物理峰值。

### 4.5 Case 1 的 128 FLOP/Byte 来源

Roofline 使用

$$
P_{\mathrm{BW}}=I\times B,
$$

其中 \(I\) 是算术强度，单位为 FLOP/Byte；\(B\) 是 GB/s。两者相乘先得到 GFLOPS，再除以 1000 得到 TFLOPS。

一个完整 K Page 包含 64 个 Token，每个 Token 有 128 个 BF16 元素，因此强制读取量为

$$
D_K=64\times128\times2
=16{,}384\text{ Bytes}.
$$

一个 Q 对该 Page 的 64 Head 点积计算量为

$$
F_{1Q}=2\times64_{\mathrm{heads}}
\times64_{\mathrm{tokens}}\times128_{\mathrm{dim}}
=1{,}048{,}576\text{ FLOPs}.
$$

Case 2 的 `next_n=1`，同一个 K Page 只服务一个 Q，所以

$$
I_{\mathrm{Case2}}
=\frac{F_{1Q}}{D_K}
=\frac{1{,}048{,}576}{16{,}384}
=64\text{ FLOP/Byte}.
$$

Case 1 的 `next_n=2`。当前框架只读取并 Packing 一次 K Page，然后让两个 Q 复用它，因此读取相同的 16 KiB K 可以完成两份点积：

$$
F_{2Q}=2\times F_{1Q}=2{,}097{,}152\text{ FLOPs},
$$

$$
I_{\mathrm{Case1}}
=\frac{F_{2Q}}{D_K}
=\frac{2{,}097{,}152}{16{,}384}
=128\text{ FLOP/Byte}.
$$

也可以从单个 BF16 K 元素理解：它占 2 Bytes；对每个 Q，它与 64 个 Head 分别执行一次乘加，即 \(64\times2=128\) FLOPs；Case 1 有两个 Q，因此每 2 Bytes K 产生 256 FLOPs，仍然是 128 FLOP/Byte。

代入随机 Page 有效带宽：

$$
P_{\mathrm{BW,Case1}}
=128\times206.25
=26{,}400\text{ GFLOPS}
=26.40\text{ TFLOPS},
$$

$$
P_{\mathrm{BW,Case2}}
=64\times206.25
=13{,}200\text{ GFLOPS}
=13.20\text{ TFLOPS}.
$$

这里的算术强度按完整 Page 和题目计分 FLOPs 估算。尾 Page 仍读取完整 16 KiB、但有效 Token 少于 64，因此真实平均算术强度会略低。更重要的是，上述带宽 Roofline 只计算不可避免的原始 K 读取，没有扣除 Packed K、Page Scores、FP32 后处理和同步，是乐观上限而非性能预测。

### 4.6 当前 Roofline 位置

结合真实 SME 峰值与随机 K Page 带宽：

$$
P_{\mathrm{roof}}
=\min(P_{\mathrm{SME}},P_{\mathrm{BW}}).
$$

| 测试用例 | SME Roof | K 带宽 Roof | 算子 Roofline | 当前无 Profiler | Roof 利用率 |
|---|---:|---:|---:|---:|---:|
| Case 1 | 38.651T | 26.40T | 26.40T | 8.929T | 33.8% |
| Case 2 | 38.651T | 13.20T | 13.20T | 10.178T | 77.1% |

Case 2 已明显进入随机 K Page 带宽受限区。13.20 TFLOPS 没有包含其它必要工作，因此当前框架更现实的目标是稳定达到约 11--12 TFLOPS，而不是接近 38.651 TFLOPS。

Case 1 的 26.40 TFLOPS 是长时间稳态下的乐观上限。实际 Case 1 只有约 128 us，每线程仅处理约 13--14 个 Page，内存和 SME 很难进入长时间稳态；Q Packing、全局 Barrier、第二个 Q 的调用以及固定启动成本占比更高，因此不能由 33.8% 的 Roof 利用率推断存在三倍可提取空间。
