#ifndef __COREFORGE_HALF_H__
#define __COREFORGE_HALF_H__

/***************************************************************************************************
 * @file Half.h
 * @brief Defines a high-performance Half-precision floating-point type (fp16) for CoreForge.
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

// IEEE 754 Half-precision floating-point type (16 bits).
struct alignas(2) Half {
  uint16_t storage;

  ///
  /// static conversion functions
  ///

  // construct Half from bitcasted uint16_t
  CF_HOST_DEVICE
  static Half bitcast(uint16_t x) {
    Half h;
    h.storage = x;
    return h;
  }

  // FP32 -> FP16 conversion - round to nearest, ties to even
  CF_HOST_DEVICE
  static Half convert(float value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return Half{__float2half_rn(value)};
#else
#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C
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
    // overflow to inf of Half
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
      // exp_fp16 <= 0 means value is too small to be represented as a normalized Half.
      // We'll try to create a subnormal Half: shift the full 24-bit significand (implicit 1 + 23 mantissa)
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
  static Half convert(int value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return Half(__int2half_rn(value));
#else
    return convert(static_cast<float>(value));
#endif
  }

  CF_HOST_DEVICE
  static Half convert(unsigned value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return Half(__uint2half_rn(value));
#else
    return convert(static_cast<float>(value));
#endif
  }

  // converts a Half-precision stored as uint16_t to a float
  CF_HOST_DEVICE
  static float convert(Half value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __half2float(*reinterpret_cast<__half*>(&value.storage));
#else
#if !defined(__CUDACC__) && COREFORGE_ENABLE_F16C
    if (detail::CpuF16CDetector::instance().isAvailable()) {
      return detail::half_to_float_f16c(value.storage);
    }
#endif

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

  Half() = default;

  struct from_bits_tag {};
  CF_HOST_DEVICE
  static constexpr from_bits_tag from_bits() { return {}; }

  CF_HOST_DEVICE
  constexpr Half(from_bits_tag, uint16_t bits) : storage(bits) {}

#if defined(__CUDACC__)
  // reinterpret cast from cuda's __half
  CF_HOST_DEVICE
  explicit Half(const __half& h) {
#if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<const uint16_t&>(h);
#else
    __half_raw hr(h);
    std::memcpy(&storage, &hr, sizeof(hr));
#endif
  }
#endif

  // float -> Half
  CF_HOST_DEVICE
  explicit Half(float value) { storage = convert(value).storage; }

  // double -> Half
  CF_HOST_DEVICE
  explicit Half(double value) {
    storage = convert(static_cast<float>(value)).storage;
  }

  // int -> Half
  CF_HOST_DEVICE
  explicit Half(int value) { storage = convert(value).storage; }

  // unsigned -> Half
  CF_HOST_DEVICE
  explicit Half(unsigned value) { storage = convert(value).storage; }

  /// assignment
#if defined(__CUDACC__)
  CF_HOST_DEVICE
  Half& operator=(const __half& h) {
#if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<const uint16_t&>(h);
#else
    __half_raw hr(h);
    std::memcpy(&storage, &hr, sizeof(hr));
#endif
    return *this;
  }
#endif

  // Half -> float
  CF_HOST_DEVICE
  operator float() const { return convert(*this); }

  // Half -> double
  CF_HOST_DEVICE
  operator double() const { return static_cast<double>(convert(*this)); }

  // Half -> int
  CF_HOST_DEVICE
  operator int() const { return static_cast<int>(convert(*this)); }

  // cast to bool
  CF_HOST_DEVICE
  explicit operator bool() const { return convert(*this) != 0.0f; }

  // bitcast to cuda's __half
#if defined(__CUDACC__)
  CF_HOST_DEVICE
  __half toCudaHalf() const {
#if defined(__CUDA_ARCH__)
    return reinterpret_cast<const __half&>(storage);
#else
    __half_raw hr;
    std::memcpy(&hr, &storage, sizeof(hr));
    return __half(hr);
#endif
  }
#endif

  // access the raw bits
  CF_HOST_DEVICE
  uint16_t raw() const { return storage; }

  // return the sign bit
  CF_HOST_DEVICE
  bool signbit() const { return (storage & 0x8000) != 0; }

  // return the biased exponent
  CF_HOST_DEVICE
  int exponent_biased() const {
    return static_cast<int>((storage >> 10) & 0x1F);
  }

  // return the unbiased exponent
  CF_HOST_DEVICE
  int exponent() const { return exponent_biased() - 15; }

  // return the mantissa
  CF_HOST_DEVICE
  int mantissa() const { return static_cast<int>(storage & 0x3FF); }
};

// for <cmath> functions
CF_HOST_DEVICE
inline bool signbit(const coreforge::Half& h) {
  return h.signbit();
}

CF_HOST_DEVICE
inline Half abs(const Half& h) {
  return Half::bitcast(h.raw() & 0x7FFF);
}

CF_HOST_DEVICE
inline bool isnan(const Half& h) {
  return (h.exponent_biased() == 0x1F) && (h.mantissa() != 0);
}

CF_HOST_DEVICE
inline bool isfinite(const Half& h) {
  return h.exponent_biased() != 0x1F;
}

CF_HOST_DEVICE
inline bool isinf(const Half& h) {
  return (h.exponent_biased() == 0x1F) && (h.mantissa() == 0);
}

CF_HOST_DEVICE
inline Half nanh(const char* tagp) {
  (void)tagp;                    // unused
  return Half::bitcast(0x7FFF);  // quiet NaN
}

CF_HOST_DEVICE
inline bool isnormal(const Half& h) {
  int exp = h.exponent_biased();
  return (exp > 0) && (exp < 0x1F);
}

// FP classification:
// FP_INFINITE, FP_NAN, FP_NORMAL, FP_SUBNORMAL, FP_ZERO
CF_HOST_DEVICE
inline int fpclassify(const Half& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x1F) {
    return (mantissa == 0) ? FP_INFINITE : FP_NAN;
  } else if (exp == 0) {
    return (mantissa == 0) ? FP_ZERO : FP_SUBNORMAL;
  } else {
    return FP_NORMAL;
  }
}

CF_HOST_DEVICE
inline Half sqrt(const Half& h) {
#if defined(__CUDA_RTC__)
  return Half(sqrtf(static_cast<float>(h)));
#else
  return Half(std::sqrt(static_cast<float>(h)));
#endif
}

// z = copysign(x, y): return a value with the magnitude of x and the sign of y
CF_HOST_DEVICE
inline Half copysign(const Half& x, const Half& y) {
  uint16_t sign = y.raw() & 0x8000;
  uint16_t mag = x.raw() & 0x7FFF;
  return Half::bitcast(sign | mag);
}
}  // namespace coreforge

#if !defined(__CUDA_RTC__)
namespace std {
// for std::numeric_limits
template <>
struct numeric_limits<coreforge::Half> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr coreforge::Half min() noexcept {
    return {coreforge::Half::from_bits(),
            0x0001};  // 2^-24, minimum positive denormalized Half
  }
  static constexpr coreforge::Half max() noexcept {
    return {coreforge::Half::from_bits(), 0x7BFF};  // (2-2^-10)*2^15
  }
  static constexpr coreforge::Half lowest() noexcept {
    return {coreforge::Half::from_bits(), 0xFBFF};  // -(2-2^-10)*2^15
  }
  static constexpr coreforge::Half infinity() noexcept {
    return {coreforge::Half::from_bits(), 0x7C00};
  }
  static constexpr coreforge::Half quiet_NaN() noexcept {
    return {coreforge::Half::from_bits(), 0x7E00};
  }
  static constexpr coreforge::Half signaling_NaN() noexcept {
    return {coreforge::Half::from_bits(), 0x7D00};
  }
  static constexpr coreforge::Half denorm_min() noexcept {
    return {coreforge::Half::from_bits(), 0x0001};  // 2^-24
  }
};
}  // namespace std
#endif  // !__CUDA_RTC__

namespace coreforge {
// artithmetic operators
CF_HOST_DEVICE
inline bool operator==(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __heq(a.toCudaHalf(), b.toCudaHalf());
#else
  return static_cast<float>(a) == static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator!=(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hne(a.toCudaHalf(), b.toCudaHalf());
#else
  return static_cast<float>(a) != static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator<(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hlt(a.toCudaHalf(), b.toCudaHalf());
#else
  return static_cast<float>(a) < static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator<=(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hle(a.toCudaHalf(), b.toCudaHalf());
#else
  return static_cast<float>(a) <= static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator>(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hgt(a.toCudaHalf(), b.toCudaHalf());
#else
  return static_cast<float>(a) > static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator>=(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hge(a.toCudaHalf(), b.toCudaHalf());
#else
  return static_cast<float>(a) >= static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline Half operator+(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return Half(__hadd(a.toCudaHalf(), b.toCudaHalf()));
#else
  return Half(static_cast<float>(a) + static_cast<float>(b));
#endif
}

CF_HOST_DEVICE
inline Half operator-(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return Half(__hsub(a.toCudaHalf(), b.toCudaHalf()));
#else
  return Half(static_cast<float>(a) - static_cast<float>(b));
#endif
}

CF_HOST_DEVICE
inline Half operator-(const Half& a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return Half(__hneg(a.toCudaHalf()));
#else
  return Half(-static_cast<float>(a));
#endif
}

CF_HOST_DEVICE
inline Half operator*(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return Half(__hmul(a.toCudaHalf(), b.toCudaHalf()));
#else
  return Half(static_cast<float>(a) * static_cast<float>(b));
#endif
}

CF_HOST_DEVICE
inline Half operator/(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return Half(__hdiv(a.toCudaHalf(), b.toCudaHalf()));
#else
  return Half(static_cast<float>(a) / static_cast<float>(b));
#endif
}

CF_HOST_DEVICE
inline Half& operator+=(Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hadd(a.toCudaHalf(), b.toCudaHalf()));
#else
  a = Half(static_cast<float>(a) + static_cast<float>(b));
#endif
  return a;
}

CF_HOST_DEVICE
inline Half& operator-=(Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hsub(a.toCudaHalf(), b.toCudaHalf()));
#else
  a = Half(static_cast<float>(a) - static_cast<float>(b));
#endif
  return a;
}

CF_HOST_DEVICE
inline Half& operator*=(Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hmul(a.toCudaHalf(), b.toCudaHalf()));
#else
  a = Half(static_cast<float>(a) * static_cast<float>(b));
#endif
  return a;
}

CF_HOST_DEVICE
inline Half& operator/=(Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hdiv(a.toCudaHalf(), b.toCudaHalf()));
#else
  a = Half(static_cast<float>(a) / static_cast<float>(b));
#endif
  return a;
}

CF_HOST_DEVICE
inline Half& operator++(Half& a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hadd(a.toCudaHalf(), Half(1.0).toCudaHalf()));
#else
  a = Half(static_cast<float>(a) + 1.0f);
#endif
  return a;
}

CF_HOST_DEVICE
inline Half operator++(Half& a, int) {
  Half ret(a);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hadd(a.toCudaHalf(), Half(1.0).toCudaHalf()));
#else
  a = Half(static_cast<float>(a) + 1.0f);
#endif
  return ret;
}

CF_HOST_DEVICE
inline Half& operator--(Half& a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hsub(a.toCudaHalf(), Half(1.0).toCudaHalf()));
#else
  a = Half(static_cast<float>(a) - 1.0f);
#endif
  return a;
}

CF_HOST_DEVICE
inline Half operator--(Half& a, int) {
  Half ret(a);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  a = Half(__hsub(a.toCudaHalf(), Half(1.0).toCudaHalf()));
#else
  a = Half(static_cast<float>(a) - 1.0f);
#endif
  return ret;
}
}  // namespace coreforge

// user-defined literal for Half-precision floating point
CF_HOST_DEVICE
inline coreforge::Half operator"" _hf(long double value) {
  return coreforge::Half(static_cast<float>(value));
}

CF_HOST_DEVICE
inline coreforge::Half operator"" _hf(unsigned long long int value) {
  return coreforge::Half(static_cast<float>(value));
}

#endif  // __COREFORGE_HALF_H__