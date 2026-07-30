#!/bin/bash

CXX=${CXX:-g++}
CPU_LIST=${CPU_LIST:-0-37}
THREADS=${OMP_NUM_THREADS:-38}
PROBE_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=$(mktemp -d /tmp/pac-roofline-probes.XXXXXX)
trap 'rm -rf "$OUT_DIR"' EXIT

CXXFLAGS=(-O3 -march=armv9-a+sme+sve2 -fopenmp)

echo "Compiler: $($CXX --version | head -n 1)"
echo "CPU_LIST=$CPU_LIST OMP_NUM_THREADS=$THREADS"

echo "===== SME peak: compile ====="
"$CXX" "${CXXFLAGS[@]}" \
    "$PROBE_DIR/probe_sme_peak.cpp" -o "$OUT_DIR/probe_sme_peak"
echo "===== SME peak: inner instructions ====="
objdump -d "$OUT_DIR/probe_sme_peak" | \
    grep -E -i 'smstart|smstop|bfmopa|zero.*za' || true
echo "===== SME peak: run ====="
taskset -c "$CPU_LIST" env \
    OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close OMP_PLACES=cores \
    "$OUT_DIR/probe_sme_peak"

echo
echo "===== Random-page bandwidth: compile ====="
"$CXX" "${CXXFLAGS[@]}" \
    "$PROBE_DIR/probe_random_page_bandwidth.cpp" \
    -lnuma -o "$OUT_DIR/probe_random_page_bandwidth"
echo "===== Random-page bandwidth: run ====="
taskset -c "$CPU_LIST" env \
    OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close OMP_PLACES=cores \
    "$OUT_DIR/probe_random_page_bandwidth"
