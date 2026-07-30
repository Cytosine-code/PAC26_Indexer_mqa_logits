# PAC 2026 — indexer_mqa_logits

## 用户偏好

1. 中文回答问题
2. 用户没有明确的代码编辑意向时不要修改代码
3. 存在强软件依赖的开发任务时，不要头铁，停下来告诉用户

## 赛题
鲲鹏芯片上优化 `indexer_bf16_paged_mqa_logits`，仅允许修改 `indexer_mqa_logits.h`。

- 精度：`cos_diff < 5e-6`
- 评分：TFLOPS
- 测试规模(固定)：
  - TC1: batch=32, next_n=2, avg_kv=1024
  - TC2: batch=128, next_n=1, avg_kv=4096

### 硬件环境

LX2芯片，详细请参考目录下SVEconfig.md，只能使用单numa节点38核
优化主力：向量化计算点积

### 本地开发

- `bash test.sh` — x86 WSL 编译运行（纯 C++ ref，调试算法逻辑）
- 远程 ARM：`g++ -O3 -march=armv9-a+sme+sve2 -fopenmp -lnuma main.cpp -o main`
- `#ifdef __aarch64__` 隔离 ARM/x86 路径，本地用 `-I./x86_compat`

### 赛题记忆

为防止你忘掉上下文如环境等配置，可以参考以下内容：

- probes/ 下保存了历史的探针
- SVEconfig.md 保存了远程计算环境的配置
- PAC2026_indexer_maq_logits_优化报告.md 记录了我们历史的优化