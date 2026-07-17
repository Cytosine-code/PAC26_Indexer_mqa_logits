#pragma once

#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <iostream>

#include "utils.h"

constexpr int64_t DEFAULT_ALIGNMENT = 64;

const int64_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

inline void *mmap_on_package_memory(int64_t size)
{
    FLASH_ASSERT(size % PAGE_SIZE == 0);
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    int64_t cpu = sched_getcpu();
    int64_t rnode = numa_node_of_cpu(cpu) + 16;
    unsigned long mask = 1UL << rnode;
    mbind(addr, size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE);
    return addr;
}

struct Allocator {
    void *begin_addr;
    void *cur_addr;
    int64_t max_size = 0;

    Allocator(int64_t max_size_)
    {
        max_size = ceil(max_size_, PAGE_SIZE);
        begin_addr = cur_addr = mmap_on_package_memory(max_size);
    }

    ~Allocator()
    {
        munmap(begin_addr, max_size);
    }

    int64_t remain()
    {
        return max_size - (reinterpret_cast<int64_t>(cur_addr) - reinterpret_cast<int64_t>(begin_addr));
    }

    void *alloc(int64_t size, int64_t alignment = DEFAULT_ALIGNMENT)
    {
        void *addr = reinterpret_cast<void *>(ceil(reinterpret_cast<int64_t>(cur_addr), alignment));
        FLASH_ASSERT((int8_t *)addr + size <= (int8_t *)begin_addr + max_size);
        cur_addr = (int8_t *)addr + size;
        return addr;
    }

    void reset()
    {
        cur_addr = begin_addr;
    }
};