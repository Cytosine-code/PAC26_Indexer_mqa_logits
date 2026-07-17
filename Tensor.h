#pragma once

#include <array>
#include <functional>

#include "allocator.h"

template <typename dtype, int64_t dim>
struct Tensor {
    static_assert(dim > 0);
public:
    Tensor() = default;

    Tensor(dtype *data_ptr, const std::array<int64_t, dim> &sizes, const std::array<int64_t, dim> &strides)
        : data_ptr_(data_ptr), sizes_(sizes), strides_(strides)
    {
    }

    Tensor(dtype *data_ptr, const int64_t *sizes) : data_ptr_(data_ptr)
    {
        for (int64_t i = dim - 1; i >= 0; --i) {
            sizes_[i] = sizes[i];
            strides_[i] = i + 1 < dim ? sizes[i + 1] * strides_[i + 1] : 1;
        }
    }

    Tensor(dtype *data_ptr, const std::array<int64_t, dim> &sizes) : data_ptr_(data_ptr), sizes_(sizes)
    {
        for (int64_t i = dim - 1; i >= 0; --i) {
            strides_[i] = i + 1 < dim ? sizes[i + 1] * strides_[i + 1] : 1;
        }
    }

    int64_t numel() const
    {
        int64_t n = 1;
        for (const auto &x : sizes_) {
            n *= x;
        }
        return n;
    }

    const std::array<int64_t, dim> &sizes() const
    {
        return sizes_;
    }

    const std::array<int64_t, dim> &strides() const
    {
        return strides_;
    }

    int64_t size(int64_t i) const
    {
        return sizes_[i < 0 ? dim + i : i];
    }

    int64_t stride(int64_t i) const
    {
        return strides_[i < 0 ? dim + i : i];
    }

    dtype *data_ptr() const
    {
        return data_ptr_;
    }

    dtype &index(const std::array<int64_t, dim> &indices) const
    {
        int64_t offset = 0;
        for (int64_t i = 0; i < dim; ++i) {
            offset += indices[i] * strides_[i];
        }
        return data_ptr_[offset];
    }

    bool is_contiguous() const
    {
        for (int i = 0; i < dim - 1; ++i) {
            if (strides_[i] != strides_[i + 1] * sizes_[i + 1]) {
                return false;
            }
        }
        return strides_[dim - 1] == 1;
    }

private:
    dtype *data_ptr_;
    std::array<int64_t, dim> sizes_;
    std::array<int64_t, dim> strides_;
};

template <typename T, int64_t N>
inline Tensor<T, N> create_tensor(Allocator &allocator, const std::array<int64_t, N> &sizes,
                                  int64_t alignment = 64)
{
    auto tensor = Tensor<T, N>((T *)allocator.alloc(0), sizes);
    allocator.alloc(tensor.numel() * sizeof(T), alignment);
    return tensor;
}

template <typename T, int64_t N>
inline Tensor<T, N> generate_tensor(Allocator &allocator, const std::array<int64_t, N> &sizes,
                                     const std::function<T()> &generator, int64_t alignment = DEFAULT_ALIGNMENT)
{
    auto tensor = create_tensor<T, N>(allocator, sizes, alignment);
    int64_t numel = tensor.numel();
    for (int64_t i = 0; i < numel; ++i) {
        tensor.data_ptr()[i] = generator();
    }
    return tensor;
}

inline bool check_mask_and_replace(const Tensor<float, 2> &a, const Tensor<float, 2> &b)
{
    int64_t rows = a.size(0);
    int64_t cols = a.size(1);
    FLASH_ASSERT(rows == b.size(0) && cols == b.size(1));

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            auto &x = a.index({i, j});
            auto &y = b.index({i, j});
            if (std::isnan(x) != std::isnan(y)) {
                return false;
            }
            if ((x == INFINITY) != (y == INFINITY)) {
                return false;
            }
            if ((x == -INFINITY) != (y == -INFINITY)) {
                return false;
            }
            if (std::isnan(x) || std::isinf(x)) {
                x = y = 0;
            }
        }
    }
    return true;
}

inline double calc_cos_diff(const Tensor<float, 2> &a, const Tensor<float, 2> &b)
{
    int64_t rows = a.size(0);
    int64_t cols = a.size(1);
    FLASH_ASSERT(rows == b.size(0) && cols == b.size(1));

    double sum_x2_y2 = 0;
    double sum_2xy = 0;
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            double x = a.index({i, j});
            double y = b.index({i, j});
            sum_x2_y2 += x * x + y * y;
            sum_2xy += 2 * x * y;
        }
    }
    if (sum_x2_y2 == 0) {
        return 0;
    }
    return 1 - sum_2xy / sum_x2_y2;
}