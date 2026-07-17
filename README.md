
# PAC 并行应用挑战赛—indexer_mqa_logits赛题
基于鲲鹏芯片实现`indexer_bf16_paged_mqa_logits`函数（见`indexer_mqa_logits.h`）。

## 编译
### 用GCC编译
```shell
source /fs_real_a800/PAC2026/HPKit26/HPKit/latest/setvars.sh --use-gcc
g++ -O3 -march=armv9-a+sme+sve2 -fopenmp -lnuma main.cpp -o main
```

### 用Bisheng编译
```shell
source /fs_real_a800/PAC2026/HPKit26/HPKit/latest/setvars.sh
clang++ -O3 -stdlib=libc++ -march=armv9-a+sme+sve2 -fopenmp -rtlib=compiler-rt -unwindlib=libunwind -lnuma main.cpp -o main
```

## 运行
### 执行命令
在当前目录下配置`test.sh`测试脚本（线程数和绑核配置供参考）：
```shell
NUM_THREADS=32
OMP_NUM_THREADS=$NUM_THREADS OMP_PROC_BIND=close taskset -c 1-$NUM_THREADS ./main
```

在当前目录下使用多队列提交测试任务：
```shell
dsub -x job -o `pwd`/out_%J.log -e `pwd`/err_%J.log -nl PAC -A PAC "bash test.sh"
```
测试任务执行时间一般在`60s`以内。

### 输出示例
```
Total Number of Testcases: 2
================================================================================
Running on Testcase 1:
Parameters: batch_size = 32, next_n = 2, avg_kv = 1024
cos_diff: xxx
Performance: xxx TFLOPS
================================================================================
Running on Testcase 2:
Parameters: batch_size = 128, next_n = 1, avg_kv = 4096
cos_diff: xxx
Performance: xxx TFLOPS
================================================================================
All Tests Ends
```
精度要求`cos_diff < 5e-6`，根据测试输出的性能结果`TFLOPS`进行评分。
```