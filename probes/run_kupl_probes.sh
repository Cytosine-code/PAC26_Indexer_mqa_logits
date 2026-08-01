#!/usr/bin/env bash
set -euo pipefail

# Run from the project root:
#   bash probes/run_kupl_probes.sh
# Override either compiler when needed:
#   MMA_CXX=clang++ PARALLEL_CXX=g++ bash probes/run_kupl_probes.sh

MMA_CXX="${MMA_CXX:-clang++}"
PARALLEL_CXX="${PARALLEL_CXX:-g++}"
MMA_BIN=/tmp/probe_kupl_mma_bf16
PARALLEL_BIN=/tmp/probe_kupl_parallel_for
OPENMP_BIN=/tmp/probe_openmp_parallel_for

if [[ ! -f indexer_mqa_logits.h || ! -f probes/probe_kupl_mma_bf16.cpp ]]; then
    echo "Run from the project root: bash probes/run_kupl_probes.sh" >&2
    exit 1
fi

echo "MMA compiler:      $($MMA_CXX --version | head -n 1)"
echo "parallel compiler: $($PARALLEL_CXX --version | head -n 1)"
echo "Existing KUPL environment:"
env | grep '^KUPL_' | sort || true

echo "===== kupl_mma_bf16: compile ====="
set -x
if "$MMA_CXX" -O3 -march=armv9-a+sme+sve2 -I. \
    probes/probe_kupl_mma_bf16.cpp \
    -o "$MMA_BIN" -lkupl -lnuma; then
    mma_compiled=1
else
    mma_compiled=0
fi
set +x
if ((mma_compiled)); then
    echo "===== kupl_mma_bf16: relevant instructions ====="
    objdump -d "$MMA_BIN" | \
        grep -E 'smstart|smstop|bfmopa|zero.*za' | head -n 80 || true
    echo "===== kupl_mma_bf16: run ====="
    set -x
    taskset -c 0 "$MMA_BIN"
    set +x
else
    echo "SKIP: KUPL MMA probe did not compile; continuing with scheduler probes"
fi

echo
echo "===== openmp_parallel_for: compile ====="
set -x
"$PARALLEL_CXX" -O3 -march=armv9-a+sme+sve2 -I. -fopenmp \
    probes/probe_openmp_parallel_for.cpp -o "$OPENMP_BIN"
set +x
echo "===== openmp_parallel_for: run ====="
set -x
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-37 "$OPENMP_BIN"
set +x

echo
echo "===== kupl_parallel_for: compile ====="
set -x
"$PARALLEL_CXX" -O3 -march=armv9-a+sme+sve2 -I. \
    probes/probe_kupl_parallel_for.cpp \
    -o "$PARALLEL_BIN" -lkupl
set +x
echo "===== kupl_parallel_for: run ====="
set -x
KUPL_EXECUTOR_COUNT=38 \
    KUPL_EXECUTOR_BACKEND=pthread \
    KUPL_SCHED_POLICY=static_mq \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores \
    taskset -c 0-37 "$PARALLEL_BIN"
set +x
