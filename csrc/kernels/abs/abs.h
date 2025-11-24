#pragma once

#include <cstddef>

namespace coreforge {
namespace kernels {
template <typename T>
void LaunchAbsKernel(const T* input, T* output, const size_t elem_num);
} // namespace kernels
} // namespace coreforge
