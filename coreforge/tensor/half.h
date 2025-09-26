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
#include <limits>
#include <cstring>
#include <ostream>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#include "config.h"

#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C \
    && (defined(__x86_64__) || defined(_M_X64) || \
        defined(__i386__) || defined(_M_IX86))
    #if defined(_MSC_VER)
        #include <immintrin.h>
        #define F16C_ROUND_NEAREST 0
    #else // GCC or Clang
        #include <x86intrin.h>
        #define F16C_ROUND_NEAREST (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
    #endif // _MSC_VER
#endif // F16C check

namespace coreforge { 
namespace detail { 
#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C \
    && (defined(__x86_64__) || defined(_M_X64) || \
        defined(__i386__) || defined(_M_IX86))
#include <cpuid.h>
class CpuF16CDetector { 
  bool available_{false};
  
  CpuF16CDetector() {
    #if defined(_MSC_VER) // MSVC
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      available_ = (cpuInfo[2] & (1 << 29)) != 0; // Check for F16C support
    #else // GCC or Clang
      unsigned int eax, ebx, ecx, edx;
      __get_cpuid(1, &eax, &ebx, &ecx, &edx);
      available_ = (ecx & (1 << 29)) != 0; // Check for F16C support
    #endif
  }
public:
  bool isAvailable() const { return available_; }
  // Singleton instance
  static CpuF16CDetector& instance() {
    static CpuF16CDetector instance;
    return instance;
  }
}; // class CpuF16CDetector 

// F16C conversion functions
inline uint16_t float_to_half_f16c(float value) { 
  #if defined(_MSC_VER)
    return _cvtss_sh(value, F16C_ROUND_NEAREST);
  #else // GCC or Clang
    return _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(value), F16C_ROUND_NEAREST));
  #endif // _MSC_VER
}

inline float half_to_float_f16c(uint16_t value) { 
  #if defined(_MSC_VER)
    return _cvtsh_ss(value);
  #else // GCC or Clang
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
  #endif // _MSC_VER
}
#endif // F16C check detail
}// namespace detail

//IEEE 754 half-precision floating-point type (16 bits).
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
  static half convert_from_float(float value) { 
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
      uint16_t exponent = static_cast<uint16_t>(((s >> 23) & 0xFF) - 127);
      int mantissa = s & 0x7FFFFF;
      uint16_t u = 0;
      
      if((s & 0x7FFFFFFF) == 0) {
        return bitcast(sign); // zero
      }

      if(exponent > 15) {
        if(exponent == 128 && mantissa != 0) {
          u = 0x7FFF; // NaN
        } else {
          u = sign | 0x7C00; // Inf 
        }
        return bitcast(u);
      }

      

  }

  half() = default;

};


}// namespace coreforge

#endif // __COREFORGE_HALF_H__