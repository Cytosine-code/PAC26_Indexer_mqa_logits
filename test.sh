#!/bin/bash
g++ -O3 -g -fopenmp main.cpp -o main
OMP_NUM_THREADS=$(nproc) ./main

# 远程测试服务器脚本内容
# source /fs_real_a800/PAC2026/HPCCKit26/HPCCKit/latest/setvars.sh --use-gcc
# g++ -O3 -march=armv9-a+sme+sve2 -fopenmp -lnuma main.cpp -o main
# NUM_THREADS=32
# OMP_NUM_THREADS=$NUM_THREADS OMP_PROC_BIND=close taskset -c 1-$NUM_THREADS ./main