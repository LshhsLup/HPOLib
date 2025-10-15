#ifndef __COREFORGE_BFLOAT16_H__
#define __COREFORGE_BFLOAT16_H__

/***************************************************************************************************
 * @file BFloat16.h
 * @brief Defines a high-performance Brain Float (bfloat16) type for CoreForge.
 *
 * This implementation prioritizes performance by using hardware intrinsics when available (CUDA)
 * and provides a robust, standards-compliant software fallback for portability.
 **************************************************************************************************/

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "config.h"

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#endif

namespace coreforge {
// IEEE 754 Brain Float (bfloat16) type (16 bits).
// 1 sign bit, 8 exponent bits, 7 mantissa bits
// float32 -> bfloat16, just truncate the lower 16 bits
struct alignas(2) BFloat16 {
  uint16_t storage;

  // construct BFloat16 from bitcasted uint16_t
  CF_HOST_DEVICE
  static BFloat16 bitcast(uint16_t x) {
    BFloat16 h;
    h.storage = x;
    return h;
  }

 private:
  struct from_32_bit_integer_tag {};
  static constexpr from_32_bit_integer_tag from_32_bit_integer() { return {}; }

  template <typename T>
  CF_HOST_DEVICE explicit BFloat16(from_32_bit_integer_tag, T value) {
    static_assert(std::is_integral<T>::value && sizeof(T) == 4,
                  "T must be a 32-bit integer type");
    float f = static_cast<float>(value);
    uint32_t as_int;
#if defined(__CUDA_ARCH__)
    as_int = reinterpret_cast<const uint32_t&>(f);
#else
    std::memcpy(&as_int, &f, sizeof(f));
#endif
    storage = static_cast<uint16_t>(as_int >> 16);
  }

 public:
  BFloat16() = default;

  // reinterpret cast from cuda's __nv_bfloat16
#if defined(__CUDACC__)
  CF_HOST_DEVICE
  explicit BFloat16(const __nv_bfloat16& h) {
    #if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<const uint16_t&>(h);
    #else
    __nv_bfloat16_raw hr(h);
    std::memcpy(&storage, &hr, sizeof(hr));
    #endif
  }
#endif

  // float -> BFloat16, round towards nearest
  CF_HOST_DEVICE
  explicit BFloat16(float value) { 
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(storage) : "f"(value));
    #else
    uint32_t bits;
    #if defined(__CUDA_ARCH__)
    bits = reinterpret_cast<const uint32_t&>(value);
    #else
    std::memcpy(&bits, &value, sizeof(value));
    #endif
    #endif
  }
};
}  // namespace coreforge

#endif