#!/bin/bash

set -u

CXX=${CXX:-g++}
CXXFLAGS=(-O2 -g -march=armv9-a+sme+sve2)
PROBE_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=$(mktemp -d /tmp/pac-arm-probes.XXXXXX)
trap 'rm -rf "$OUT_DIR"' EXIT

run_probe()
{
    local name=$1
    local source=$2
    echo "===== $name: compile ====="
    if ! "$CXX" "${CXXFLAGS[@]}" "$PROBE_DIR/$source" -o "$OUT_DIR/$name"; then
        echo "FAIL: $name did not compile"
        return
    fi

    echo "===== $name: relevant instructions ====="
    objdump -d "$OUT_DIR/$name" | grep -E -i 'bfdot|bfmopa|smstart|smstop|cntw|zero.*za|mova|ld1w|st1w' || true
    echo "===== $name: run ====="
    "$OUT_DIR/$name"
    echo
}

echo "Compiler: $($CXX --version | head -n 1)"
run_probe arm_features probe_arm_features.cpp
run_probe sve_bf16 probe_sve_bf16.cpp
run_probe sme_bfmopa probe_sme_bfmopa.cpp
run_probe sme_k_transpose probe_sme_k_transpose.cpp
