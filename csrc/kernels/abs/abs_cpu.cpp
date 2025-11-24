#include "abs.h"

template <typename T>
void CPUAbsImpl(const T* input, T* output, const size_t elem_num) {
  for(size_t i = 0; i < elem_num; ++i) {
    output[i] = std::abs(input[i]);
  }
}

template <typename T>
void LaunchAbsKernel(const T* input, T* output, const size_t elem_num) {
  CPUAbsImpl(input, output, elem_num);
}