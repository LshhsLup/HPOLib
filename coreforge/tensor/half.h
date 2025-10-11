#ifndef __COREFORGE_HALF_H__
#define __COREFORGE_HALF_H__

/***************************************************************************************************
 * @file half.h
 * @brief Defines a high-performance half-precision floating-point type (fp16) for CoreForge.
 *
 * This implementation prioritizes performance by using hardware intrinsics when available (CUDA, F16C)
 * and provides a robust, standards-compliant software fallback for portability.
 **************************************************************************************************/

//====================================
// Configure CPU hardware acceleration
//====================================

#ifndef COREFORGE_ENABLE_F16C
#define COREFORGE_ENABLE_F16C 1
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#include "config.h"

#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C &&                \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86))
#if defined(_MSC_VER)
#include <immintrin.h>
#define F16C_ROUND_NEAREST 0
#else  // GCC or Clang
#include <x86intrin.h>
#define F16C_ROUND_NEAREST (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#endif  // _MSC_VER
#endif  // F16C check

namespace coreforge {
namespace detail {
#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C &&                \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86))
#include <cpuid.h>
class CpuF16CDetector {
  bool available_{false};

  CpuF16CDetector() {
#if defined(_MSC_VER)  // MSVC
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    available_ = (cpuInfo[2] & (1 << 29)) != 0;  // Check for F16C support
#else                                            // GCC or Clang
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    available_ = (ecx & (1 << 29)) != 0;  // Check for F16C support
#endif
  }

 public:
  bool isAvailable() const { return available_; }
  // Singleton instance
  static CpuF16CDetector& instance() {
    static CpuF16CDetector instance;
    return instance;
  }
};  // class CpuF16CDetector

// F16C conversion functions
inline uint16_t float_to_half_f16c(float value) {
#if defined(_MSC_VER)
  return _cvtss_sh(value, F16C_ROUND_NEAREST);
#else   // GCC or Clang
  return _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(value), F16C_ROUND_NEAREST));
#endif  // _MSC_VER
}

inline float half_to_float_f16c(uint16_t value) {
#if defined(_MSC_VER)
  return _cvtsh_ss(value);
#else   // GCC or Clang
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
#endif  // _MSC_VER
}
#endif  // F16C check detail
}  // namespace detail

// IEEE 754 half-precision floating-point type (16 bits).
struct alignas(2) half {
  uint16_t storage;

  ///
  /// static conversion functions
  ///

  // construct half from bitcasted uint16_t
  CF_HOST_DEVICE
  static half bitcast(uint16_t x) {
    half h;
    h.storage = x;
    return h;
  }

  // FP32 -> FP16 conversion - round to nearest, ties to even
  CF_HOST_DEVICE
  static half convert(float value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half{__float2half_rn(value)};
#else
#if !defined(__CUDA_ARCH__) && COREFORGE_ENABLE_F16C
    if (detail::CpuF16CDetector::instance().isAvailable()) {
      uint16_t hbits = detail::float_to_half_f16c(value);
      return bitcast(hbits);
    }
#endif

    // software implementation rounds toward nearest even
    uint32_t s;
    std::memcpy(&s, &value, sizeof(value));
    uint16_t sign = static_cast<uint16_t>((s >> 16) & 0x8000);
    uint16_t exp_fp32 = static_cast<uint16_t>((s >> 23) & 0xFF);
    int16_t exp_val = exp_fp32 - 127;
    int mantissa = s & 0x7FFFFF;
    uint16_t u = 0;

    if ((s & 0x7FFFFFFF) == 0) {
      return bitcast(sign);  // zero
    }

    if (exp_fp32 == 0xFF) {  // Inf or NaN
      if (mantissa == 0) {
        u = sign | 0x7C00;  // Inf
      } else {
        u = 0x7fff;  // NaN
      }
      return bitcast(u);
    }

    // convert exponent
    // exp_val = exp_fp32 - 127
    // exp_val = exp_fp16 - 15
    // exp_fp16 = exp_fp32 - 112
    int16_t exp_fp16 = exp_val + 15;
    // overflow to inf of half
    if (exp_fp16 >= 0x1F) {
      u = sign | 0x7C00;
      return bitcast(u);
    }

    int sticky_bit = 0;  // for rounding
    if (exp_fp16 > 0) {
      // normalized
      u = static_cast<uint16_t>((exp_fp16 & 0x1F) << 10);  // fp16 exponent
      u = static_cast<uint16_t>(u | (mantissa >> 13));     // fp16 mantissa
    } else {
      // denormalized or underflow to zero
      // exp_fp16 <= 0 means value is too small to be represented as a normalized half.
      // We'll try to create a subnormal half: shift the full 24-bit significand (implicit 1 + 23 mantissa)
      // right by (rshift) so that the exponent effectively becomes -14.
      // fp32:  value = (-1)^sign * 1.mantissa * 2^(exp_val)
      // fp16:  value = (-1)^sign * 0.mantissa * 2^(-14)
      // (-1)^sign * 1.mantissa * 2^(exp_val) == (-1)^sign * 0.mantissa * 2^(-14)
      // 1.mantissa * 2^(exp_val) == 0.mantissa * 2^(-14)
      // exp_val < -14, so we need to right shift the mantissa by (-14 - exp_val) to make the exponents equal -14.
      // eg. exp_val = -15, we need to shift right by 1 to make exponent -14.
      // 0.1mantissa * 2^(-14) = 1.mantissa * 2^(-15) = 0.mantissa * 2^(-14)
      int rshift = -14 - exp_val;
      if (rshift < 24) {
        mantissa |= (1 << 23);  // add implicit leading 1
        // check rshift bits are all zero, if not, set sticky bit, for rounding
        sticky_bit = (mantissa & ((1 << rshift) - 1)) != 0;
        mantissa = mantissa >> rshift;
        u = (static_cast<uint16_t>(mantissa >> 13) & 0x3FF);  // fp16 mantissa
      } else {
        // underflow to zero
        mantissa = 0;
        u = 0;
      }
    }

    // round to nearest even
    int round_bit = (mantissa >> 12) & 0x1;
    sticky_bit |= (mantissa & ((1 << 12) - 1)) != 0;
    if (round_bit && sticky_bit || round_bit && (u & 0x1)) {
      // 1. round bit is 1 and sticky bit is 1, must be > .5, round up
      // 2. round bit is 1 and sticky bit is 0, exactly .5, round to even
      //    (u & 0x1) is 1, so original number is odd, round up to even
      u = static_cast<uint16_t>(u + 1);
    }
    u |= sign;
    return bitcast(u);
    // convert
#endif
  }

  // FP32 -> FP16 conversion - round to nearest, ties to even
  CF_HOST_DEVICE
  static half convert(int value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half(__int2half_rn(value));
#else
    return convert(static_cast<float>(value));
#endif
  }

  CF_HOST_DEVICE
  static half convert(unsigned value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half(__uint2half_rn(value));
#else
    return convert(static_cast<float>(value));
#endif
  }

  // converts a half-precision stored as uint16_t to a float
  CF_HOST_DEVICE
  static float convert(half value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __half2float(*reinterpret_cast<__half*>(&value.storage));
#else
#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C
    if (detail::CpuF16CDetector::instance().isAvailable()) {
      return detail::half_to_float_f16c(value.storage);
    }

    // software implementation
    const uint16_t h = value.storage;
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp_fp16 = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    unsigned f = 0;

    if (exp_fp16 > 0 && exp_fp16 < 31) {  // normalized
      // convert exponent
      // exp_val = exp_fp16 - 15
      // exp_val = exp_fp32 - 127
      // exp_fp32 = exp_fp16 + 112
      uint32_t exp_fp32 = exp_fp16 + 112;
      f = (sign << 31) | (exp_fp32 << 23) | (mantissa << 13);
    } else if (exp_fp16 == 0) {  // denormalized or zero
      if (mantissa) {
        // denormalized
        // value_fp16 = (-1)^sign * 0.mantissa_fp16 * 2^(-14)
        // value_fp32 = (-1)^sign * 1.mantissa_fp32 * 2^(exp_fp32 - 127)
        // 0.mantissa_fp16 * 2^(-14) = 1.mantissa_fp32 * 2^(exp_fp32 - 127)
        // we need to find exp_fp32 and mantissa_fp32
        uint32_t exp_fp32 =
            exp_fp16 + 113;  // start with exp_fp32 = -14 + 127 = 113
        // 0.xxxxxxxxxx ==> 1.yyyyyyyyyyyyyyyyyyyyyyy lshift, so decrement exp_fp32
        while ((mantissa & (1 << 10)) == 0) {
          mantissa <<= 1;
          exp_fp32--;
        }
        // we need to remove the leading 1 for mantissa_fp32
        mantissa &= 0x3FF;  // remove leading 1
        f = (sign << 31) | (exp_fp32 << 23) | (mantissa << 13);
      } else {
        // zero
        f = sign << 31;
      }
    } else if (exp_fp16 == 31) {  // Inf or NaN
      if (mantissa == 0) {
        f = (sign << 31) | (0xFF << 23);  // Inf
      } else {
        f = 0x7FFFFFFF;  // NaN
      }
    }
#if defined(__CUDA_ARCH__)
    return __uint_as_float(f);
#else
    float result;
    std::memcpy(&result, &f, sizeof(f));
    return result;
#endif
#endif
  }

  half() = default;

#if defined(__CUDACC__)
  // reinterpret cast from cuda's __half
  CF_HOST_DEVICE
  half(const __half& h) {
#if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<const uint16_t&>(h);
#else
    __half_raw hr(h);
    std::memcpy(&storage, &hr, sizeof(hr));
#endif
  }
#endif
};

}  // namespace coreforge

#endif  // __COREFORGE_HALF_H__