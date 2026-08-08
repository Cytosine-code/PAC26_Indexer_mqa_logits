# PAC 2026 indexer_mqa_logits 优化汇报 PPT 文案

---

## 第 1 页：赛题分析 - DeepSeek Indexer 中的 MQA Logits

### 左侧：算子背景

**应用场景**

- 算子服务于 DeepSeek Indexer，在长上下文中计算 Query 与历史 Key Token 的相关性，为后续 Token 筛选和稀疏注意力提供索引分数。
- 采用 MQA（Multi-Query Attention）结构：64 个 Query Head 共享一组 Key Cache，降低长序列 KV Cache 的存储和访存开销。
- Key 按 64 Token Page 分页存储，通过 Block Table 完成逻辑 Page 到物理 Page 的映射。
- 输入 Q/K 为 BF16，点积累加、Head 权重和最终 Logits 为 FP32；无效 Token 输出 `-inf`。

**固定测试规模**

| 用例 | Batch | next_n | 平均 KV 长度 | Heads | Head Dim | Page |
|---|---:|---:|---:|---:|---:|---:|
| Case 1 | 32 | 2 | 1024 | 64 | 128 | 64 Token |
| Case 2 | 128 | 1 | 4096 | 64 | 128 | 64 Token |

### 右侧：公式化表达

对 Batch `b`、新增 Query `n`、Head `h` 和有效历史 Token `t`：

$$
S_{b,n,h,t}
=\sum_{d=0}^{D-1}
Q_{b,n,h,d}\,K_{b,t,d}
$$

对每个 Head 的点积先执行 ReLU，再进行带权 Head Reduction：

$$
O_{b,n,t}
=\sum_{h=0}^{H-1}
W_{b,n,h}\cdot\max(S_{b,n,h,t},0)
$$

$$
O_{b,n,t}=-\infty,\qquad t\ge L_{b,n}
$$

其中固定参数为：

$$
H=64,\qquad D=128,\qquad T_{page}=64
$$

**计算特点**

- 主计算为大量长度 128 的 BF16 点积，适合 SVE/SME 向量与矩阵指令。
- 同一个 K Page 被 64 个 Query Head 使用；Case 1 中还被两个 Q 复用。
- ReLU 位于点积与 Head Reduction 之间，不能将两层线性计算直接合并。
- 随机物理 Page、数据 Packing、短任务同步和 NUMA 亲和性都会影响最终性能。

---

## 第 2 页：硬件分析 - LX2 单 NUMA 的真实能力

### 计算节点

| 项目 | 实测配置 |
|---|---:|
| 架构 | AArch64 / LX2 |
| 单 NUMA 计算核心 | 38 核，CPU `0-37` |
| SMT | 无 |
| 最高主频 | 2.0 GHz |
| L1 Data / Instruction | 32 KiB / 32 KiB 每核 |
| L2 | 768 KiB 每核 |
| Cache Line | 64 Byte |

**NUMA 结论**

- 运行必须严格绑定单 NUMA 的 CPU `0-37`。
- 曾使用 `taskset -c 1-38`，仅一个核心跨 NUMA 就使性能从约 9 TFLOPS 降至 6-7 TFLOPS。
- NUMA、绑核和内存归属不是外围配置，而是算子性能模型的一部分。

### SVE / SME 能力

| 能力 | 实测结果 |
|---|---:|
| SVE / SME Vector Length | 512 bit（64 Byte） |
| SVE BF16 lanes | 32 |
| SVE FP32 lanes | 16 |
| SVE BF16 指令 | `BFDOT`，64 FLOPs/条 |
| SME FP32 ZA Tile | `16 x 16` |
| SME BF16 指令 | `BFMOPA`，1024 FLOPs/条 |
| 38 核纯 BFMOPA 峰值 | 38.651 TFLOPS |
| 真实内核指令组合上限 | 34.945 TFLOPS |

### 带宽探针

| Page 访问模式 | 有效带宽 |
|---|---:|
| 顺序 Page + 线性 Cache Line | 398.8 GB/s |
| 随机 Page + 线性 Cache Line | 356.4 GB/s |
| 顺序 Page + K Packing 跨行序 | 357.5 GB/s |
| 随机 Page + K Packing 跨行序 | 209.6 GB/s |
| 生产复刻：随机 + 跨行 + 预取 1 Page | 228.3 GB/s |

**硬件分析结论**

- 随机 Page 映射本身代价有限，真正的带宽损失来自“随机 Page + Page 内 256 Byte 跨行访问”的组合。
- SVE 适合点积和 FP32 后处理，SME 适合 Page 级矩阵乘与 ZA 转置。
- 优化目标不是单纯逼近计算峰值，而是同时处理矩阵吞吐、Packing、带宽和并行长尾。

---

## 第 3 页：优化技术一 - Page 矩阵化：从点积循环到 GEMM

### 原始计算

参考实现逐 Batch、Query、Head、Token 执行长度 128 的点积：

```text
for batch
  for query
    for head
      for token
        for dim
          dot += Q[head, dim] * K[token, dim]
```

每个点积都需要独立的循环控制和 FP32 横向归约，无法充分利用矩阵单元。

### Page 级矩阵表达

对一个 64 Token Page：

$$
Q\in\mathbb{R}^{H\times D},\qquad
K^T\in\mathbb{R}^{D\times T}
$$

第一阶段：一次矩阵乘生成所有 Head-Token Scores。

$$
S_{H\times T}=Q_{H\times D}K^T_{D\times T}
$$

第二阶段：ReLU 与 Head 权重归约。

$$
O_{1\times T}=W_{1\times H}\operatorname{ReLU}(S_{H\times T})
$$

即：

$$
O_t=\sum_h W_h\max\left(\sum_dQ_{h,d}K_{t,d},0\right)
$$

### 严格等价性

- 每个 `S[h,t]` 仍由同一组 128 个 BF16 乘积以 FP32 累加得到。
- ReLU 仍逐 Head、逐 Token 执行，没有跨越非线性操作。
- Head Reduction 顺序保持一致，Mask 语义不变。
- 两个用例最终 `cos_diff` 约为 `2.2e-14 / 5.0e-14`，远低于 `5e-6` 精度要求。

### 优化价值

- 将大量小点积转换为规则的 Page 矩阵块。
- 一次 BFMOPA 同时更新 `16 x 16` 个 FP32 Score，消除逐点横向归约。
- Q 每个 Query 只 Packing 一次并跨所有 Page 复用；K 每 Page 只 Packing 一次并被全部 Head 使用。
- SVE 版本约 `0.64/0.73 TFLOPS`，接入 SME Page 矩阵化和 Packing 后跃升至 `7.80/6.60 TFLOPS`。

---

## 第 4 页：优化技术二 - SME/SVE 协同内核与数据布局

### SME BFMOPA 主内核

在 512-bit SME 下：

$$
ZA[16,16]\mathrel{+}=A[16,2]\times B[2,16]
$$

一条 `BFMOPA` 完成：

$$
16\times16\times2\ \text{MAC}=1024\ \text{FLOPs}
$$

完整 Page 被分解为 `4 Head Tile x 4 Token Tile`，K 维 128 被分为 64 个 BF16 Pair。

### 2x2 ZA 寄存器分块

四个 ZA Tile 同时计算两个 Head Block 和两个 Token Block：

```text
ZA0 = Head A x Token A
ZA1 = Head A x Token B
ZA2 = Head B x Token A
ZA3 = Head B x Token B
```

每个 K Pair 的向量 Load：

$$
4Q+1K\quad\longrightarrow\quad2Q+2K
$$

完整 Page 输入 Load 数量：

$$
1280\quad\longrightarrow\quad1024,\qquad -20\%
$$

同机实测提升：Case 1 `+4.76%`，Case 2 `+7.77%`。

### ZA 转置加速 K Packing

原 SVE Gather 从 16 个相隔 256 Byte 的 Token Row 收集 BF16 Pair。优化后把 K Page 划分为 `16 x 16 uint32` Tile：

```asm
ld1w {za0h.s[...]}, ...   // 横向连续加载 16 行
st1w {za0v.s[...]}, ...   // 从纵向 Slice 转置写出
```

- 直接利用 ZA 完成位模式转置，不做数值转换。
- 热 Page 探针由 `2967.6 ns/page` 降至 `619.6 ns/page`，Packing 微内核加速 `4.79x`。

### SVE 辅助路径

- Q Packing：16-lane 32-bit Gather，一次收集 16 个 Head 的 BF16 Pair。
- 尾 Page：SVE Predicate 保证边界安全。
- FP32 后处理：SVE 完成 ReLU、Weight 和 Head Reduction。
- 保留 16 KiB `page_scores` 作为 SME 与 SVE 的 L1 解耦缓冲；实测直接消除该缓冲反而显著降速。

---

## 第 5 页：优化技术三 - 从 Batch 并行到 Page 级均衡调度

### 原有问题：按 Batch 静态分配

第 `b` 个 Batch 的 Page 数：

$$
N_b=\left\lceil\frac{L_b}{64}\right\rceil
$$

线程负载：

$$
W_t\propto\sum_{b\in\mathcal{B}_t}N_b
$$

总耗时由最慢线程决定：

$$
T_{parallel}\approx\max_tW_t
$$

- Case 1 只有 32 个 Batch，无法使用全部 38 核。
- Batch 的 Context Length 不同，Case 2 也存在 Page 数量长尾。

### Page 前缀和与等量切分

构造 Page 前缀和：

$$
P_0=0,\qquad P_{b+1}=P_b+N_b
$$

线程 `t` 处理：

$$
s_t=\left\lfloor\frac{P_Bt}{T}\right\rfloor,\qquad
e_t=\left\lfloor\frac{P_B(t+1)}{T}\right\rfloor
$$

保证任意线程 Page 数最多相差 1：

$$
\max_t(e_t-s_t)-\min_t(e_t-s_t)\le1
$$

| 用例 | 总 Page 数 | 38 核每线程 Page 数 |
|---|---:|---:|
| Case 1 | 约 512 | 13-14 |
| Case 2 | 约 8192 | 215-216 |

### 调度框架设计

- 单个 OpenMP Parallel Region：先并行 Q Packing，再处理连续 Page 区间。
- 共享 Packed Q Workspace 跨 Page 复用；`packed_k` 和 `page_scores` 保持线程私有。
- 每个线程只在区间起点执行一次 `upper_bound`，随后按 Batch-major 顺序线性推进。
- 无 Task、无动态队列、无 Page 级原子竞争。
- 配合严格 `0-37` 绑核和 NUMA 内存绑定，使 38 核全部稳定参与有效计算。

**结果**

- Case 1 突破原先 32 Batch 的并行度上限。
- Case 2 基本消除 Context Length 导致的长尾。
- V9 到 V10：约 `8.46/7.55` 提升至 `8.6/9.5 TFLOPS`，Case 2 再次超过 Case 1。

---

## 第 6 页：优化技术四 - 访存、计算与后处理流水

### 1. 在 SME 计算窗口预取下一 K Page

早期方案在 K Packing 前预取，与当前 Page 的 DRAM 读取竞争，收益为零或负值。

V12 将下一 Page 的预取放入当前 Page 的 BFMOPA 循环：

```text
当前 Page：Packed K -> BFMOPA
                         ||
下一 Page：256 x PRFM，按 64 Byte 线性覆盖完整 16 KiB
```

```asm
bfmopa ...
prfm pldl2keep, [next_k]
add next_k, next_k, #64
```

- BFMOPA 使用矩阵单元时，Load/Store 与 DRAM 通道用于拉取下一 Page。
- 下一轮 K Packing 从突发 DRAM 读取转为缓存内 ZA 转置。
- Case 2 K Packing 由约 `0.47 ms` 降至 `0.18 ms`。
- V10x 到 V12：Case 1 `+9.7%`，Case 2 `+41.3%`。

### 2. 融合 K Packing 与第一个有效 Q

```text
优化前：smstart -> K Pack -> smstop -> smstart -> BFMOPA -> smstop
优化后：smstart -> K Pack -> BFMOPA -> smstop
```

消除一次 Streaming Mode 往返、函数序言和寄存器保存恢复；使用 `packed_k_ready` 保证尾 Page 的第一个有效 Q 也能正确触发融合。

### 3. 四 Token Tile 联合后处理

满 Page 同时维护 4 个 SVE FP32 累加器：

$$
R_i\mathrel{+}=W_h\cdot\max(S_{h,i},0),\qquad i=0,1,2,3
$$

- Head 权重广播由 `4 x 64=256` 次减少到 64 次，降低 75%。
- 四条独立 FMA 链提高指令级并行度。
- 尾 Page 继续使用 Predicate 路径，保证边界正确。

### 4. Q-ready 消除全局 Barrier

```text
omp for nowait
Q row Pack 完成 -> release 标志
Page worker     -> acquire 对应 Batch 的 Q rows
```

- 线程无需等待最慢的 Q Packing Worker。
- 每 Batch 只等待一次；64 Byte stride 避免 Ready Flag 伪共享。
- Case 1 中 12 个只负责一行 Q 的线程可提前进入 Page 计算，实测约 `+0.4 TFLOPS`。

---

## 第 7 页：优化效果 - 从 0.04T 到 10T/14T

### 性能演进数据（用于双折线图，纵轴建议使用对数坐标）

| 版本 | 核心优化 | Case 1 TFLOPS | Case 2 TFLOPS |
|---|---|---:|---:|
| Baseline | 参考五层循环 | 0.0382 | 0.0424 |
| V2 | SVE BF16 BFDOT | 0.641 | 0.730 |
| V4 | SME Page 矩阵化 + SVE K Packing | 7.799 | 6.600 |
| V6 | SVE Q Packing | 8.025 | 6.794 |
| V9 | SME ZA K Packing | 8.463 | 7.554 |
| V10 | Page 级均衡调度 + 38 核 | 8.600 | 9.500 |
| V10x | 融合、联合后处理、2x2 BFMOPA | 9.200 | 10.178 |
| V12 | SME 期间预取下一 K Page | 10.200 | 14.700 |
| **V12 定稿** | **Q-ready 消除全局 Barrier** | **10.500** | **14.700** |

> 实验性 `K -> scratch -> ZA` 线性 Copy 曾达到约 `10.5/14.8 TFLOPS`，但提升微弱且增加 L1 流量与实现复杂度，不计入定稿主线。

### 最终结果

| 指标 | Case 1 | Case 2 |
|---|---:|---:|
| Baseline | 38.219666 GFLOPS | 42.419292 GFLOPS |
| 最终性能 | 10.5 TFLOPS | 14.7 TFLOPS |
| 平均耗时 | 约 101 us | 约 582 us |
| 累计加速 | **274.7x** | **346.5x** |
| `cos_diff` | 约 `2.2e-14` | 约 `5.0e-14` |

### 总结语

**以 Page 矩阵化重构算法，以 SME/SVE 榨取计算吞吐，以 Page 级调度释放 38 核并行度，再通过计算期间预取实现访存与矩阵计算重叠。最终在满足 `cos_diff < 5e-6` 的前提下，实现最高约 346 倍加速。**

