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
#include <limits>

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

  struct from_bits_tag {};
  static constexpr CF_HOST_DEVICE from_bits_tag from_bits() { return {}; }

  constexpr CF_HOST_DEVICE BFloat16(from_bits_tag, uint16_t bits) : storage(bits) {}

 private:
  template <typename T>
  CF_HOST_DEVICE explicit BFloat16(from_bits_tag, T value) {
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
    (__CUDACC_VER_MAJOR__ >= 11)
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(storage) : "f"(value));
#else
    uint32_t bits;
#if defined(__CUDA_ARCH__)
    bits = reinterpret_cast<const uint32_t&>(value);
#else
    std::memcpy(&bits, &value, sizeof(value));
#endif
    if ((bits & 0x7F800000) != 0x7F800000) {
      // not Inf or NaN
      bool mantissa_bit = (bits & (1 << 16)) != 0;
      bool round_bit = (bits & (1 << 15)) != 0;
      bool sticky_bit = (bits & ((1 << 15) - 1)) != 0;
      if (round_bit && (mantissa_bit || sticky_bit)) {
        bits += static_cast<uint32_t>(1 << 16);  // round up
      }
    } else if ((bits & 0x007FFFFF) != 0) {
      // NaN
      bits = 0x7FFFFFFF;
    }
    storage = static_cast<uint16_t>(bits >> 16);
#endif
  }

  // double -> BFloat16
  CF_HOST_DEVICE
  explicit BFloat16(double value) : BFloat16(static_cast<float>(value)) {}

  // int -> BFloat16
  // wht not use BFloat16(static_cast<float>(value))?
  // Because it may lose precision when value > 2^24
  CF_HOST_DEVICE
  explicit BFloat16(int value) : BFloat16(from_bits(), value) {}

  // uint32_t -> BFloat16
  CF_HOST_DEVICE
  explicit BFloat16(uint32_t value) : BFloat16(from_bits(), value) {}

  // BFloat16 -> float
  CF_HOST_DEVICE
  operator float() const {
    uint32_t as_int = static_cast<uint32_t>(storage << 16);
#if defined(__CUDA_ARCH__)
    return *reinterpret_cast<float*>(&as_int);
#else
    float result;
    std::memcpy(&result, &as_int, sizeof(result));
    return result;
#endif
  }

  // BFloat16 -> double
  CF_HOST_DEVICE
  operator double() const { return static_cast<double>(static_cast<float>(*this)); }

  // BFloat16 -> int
  CF_HOST_DEVICE
  operator int() const { return static_cast<int>(static_cast<float>(*this)); }  

  // cast to bool
  CF_HOST_DEVICE
  explicit operator bool() const { return static_cast<float>(*this) != 0.0f; }

  // bitcast to cuda's __nv_bfloat16
#if defined(__CUDACC__)
  CF_HOST_DEVICE
  __nv_bfloat16 toCudaBFloat16() const {
#if defined(__CUDA_ARCH__)
    return reinterpret_cast<const __nv_bfloat16&>(storage);
#else 
    __nv_bfloat16_raw hr;
    std::memcpy(&hr, &storage, sizeof(hr));
    return __nv_bfloat16(hr);
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
  int exponent_biased() const { return static_cast<int>(raw() >> 7) & 0xFF; }

  // return the unbiased exponent
  CF_HOST_DEVICE
  int exponent() const { return exponent_biased() - 127; }

  // return the mantissa
  CF_HOST_DEVICE
  int mantissa() const { return static_cast<int>(raw() & 0x7F); }
};

// for cmath functions
CF_HOST_DEVICE
inline bool signbit(const BFloat16& h) {
  return h.signbit();
}

// abs
CF_HOST_DEVICE
inline BFloat16 abs(const BFloat16& h) {
  return BFloat16::bitcast(h.raw() & 0x7FFF);
}

// isnan
CF_HOST_DEVICE
inline bool isnan(const BFloat16& h) {
  return (h.exponent_biased() == 0xFF) && (h.mantissa() != 0);
}

// isfinite
CF_HOST_DEVICE
inline bool isfinite(const BFloat16& h) {
  return h.exponent_biased() != 0xFF;
}

// nan_bf16
CF_HOST_DEVICE
inline BFloat16 nan_bf16(const char* tagp) {
  (void)tagp;                      // unused
  return BFloat16::bitcast(0x7FFF);  // quiet NaN
}

// isinf
CF_HOST_DEVICE
inline bool isinf(const BFloat16& h) {
  return (h.exponent_biased() == 0xFF) && (h.mantissa() == 0);
}

// isnormal
CF_HOST_DEVICE
inline bool isnormal(const BFloat16& h) {
  int exp = h.exponent_biased();
  return (exp > 0) && (exp < 0xFF);
}

// FP classification:
// FP_INFINITE, FP_NAN, FP_NORMAL, FP_SUBNORMAL, FP_ZERO
CF_HOST_DEVICE
inline int fpclassify(const BFloat16& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0xFF) {
    return (mantissa == 0) ? FP_INFINITE : FP_NAN;
  } else if (exp == 0) {
    return (mantissa == 0) ? FP_ZERO : FP_SUBNORMAL;
  } else {
    return FP_NORMAL;
  }
}

// sqrt
CF_HOST_DEVICE
inline BFloat16 sqrt(const BFloat16& h) { 
#if defined(__CUDA_RTC__)
  return BFloat16(sqrtf(static_cast<float>(h)));
#else
  return BFloat16(std::sqrt(static_cast<float>(h)));
#endif
}

// copysign
CF_HOST_DEVICE
inline BFloat16 copysign(const BFloat16& x, const BFloat16& y) {
  uint16_t x_bits, y_bits;
#if defined(__CUDA_ARCH__)
  x_bits = reinterpret_cast<const uint16_t&>(x);
  y_bits = reinterpret_cast<const uint16_t&>(y);
#else
  std::memcpy(&x_bits, &x, sizeof(x));
  std::memcpy(&y_bits, &y, sizeof(y));
#endif
  uint16_t a_magnitude = x_bits & 0x7FFF;
  uint16_t b_sign = y_bits & 0x8000;
  return BFloat16::bitcast(a_magnitude | b_sign);
}
}  // namespace coreforge

#if !defined(__CUDA_RTC__)
namespace std {
// std::numeric_limits specialization for BFloat16
template <>
struct numeric_limits<coreforge::BFloat16> {
   static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

  static constexpr coreforge::BFloat16 min() {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x0080)};
  }

  static constexpr coreforge::BFloat16 max()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x7F7F)};
  }

  static constexpr coreforge::BFloat16 epsilon()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x0001)};
  }

  static constexpr coreforge::BFloat16 round_error()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x0001)};
  }

  static constexpr coreforge::BFloat16 infinity()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x7F80)};
  }

  static constexpr coreforge::BFloat16 quiet_NaN()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x7FC0)};
  }

  static constexpr coreforge::BFloat16 signaling_NaN()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x7FA0)};
  }

  static constexpr coreforge::BFloat16 denorm_min()  {
    return {coreforge::BFloat16::from_bits(), static_cast<uint16_t>(0x0001)};
  }
};
} // namespace std
#endif // !defined(__CUDA_RTC__)

namespace coreforge {
// arthmetic operators
CF_HOST_DEVICE
inline bool operator==(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return __heq(a.toCudaBFloat16(), b.toCudaBFloat16());
#else
  return static_cast<float>(a) == static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator!=(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return __hne(a.toCudaBFloat16(), b.toCudaBFloat16());
#else
  return static_cast<float>(a) != static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator<(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return __hlt(a.toCudaBFloat16(), b.toCudaBFloat16());
#else
  return static_cast<float>(a) < static_cast<float>(b); 
#endif
}

CF_HOST_DEVICE
inline bool operator<=(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return __hle(a.toCudaBFloat16(), b.toCudaBFloat16());
#else
  return static_cast<float>(a) <= static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator>(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return __hgt(a.toCudaBFloat16(), b.toCudaBFloat16());
#else
  return static_cast<float>(a) > static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline bool operator>=(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return __hge(a.toCudaBFloat16(), b.toCudaBFloat16());
#else
  return static_cast<float>(a) >= static_cast<float>(b);
#endif
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator+(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return coreforge::BFloat16(__hadd(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  return coreforge::BFloat16(static_cast<float>(a) + static_cast<float>(b));
#endif
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator-(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return coreforge::BFloat16(__hsub(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  return coreforge::BFloat16(static_cast<float>(a) - static_cast<float>(b));  
#endif
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator*(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return coreforge::BFloat16(__hmul(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  return coreforge::BFloat16(static_cast<float>(a) * static_cast<float>(b));  
#endif
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator/(const coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return coreforge::BFloat16(__hdiv(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  return coreforge::BFloat16(static_cast<float>(a) / static_cast<float>(b));  
#endif
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator-(const coreforge::BFloat16& a) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  return coreforge::BFloat16(__hneg(a.toCudaBFloat16()));
#else
  return static_cast<coreforge::BFloat16>(-static_cast<float>(a));
#endif
}

CF_HOST_DEVICE
inline coreforge::BFloat16& operator+=(coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hadd(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) + static_cast<float>(b));
#endif
  return a;
}

CF_HOST_DEVICE
inline coreforge::BFloat16& operator-=(coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hsub(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) - static_cast<float>(b)); 
#endif
  return a;
}

CF_HOST_DEVICE
inline coreforge::BFloat16& operator*=(coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hmul(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) * static_cast<float>(b)); 
#endif
  return a;
}

CF_HOST_DEVICE
inline coreforge::BFloat16& operator/=(coreforge::BFloat16& a, const coreforge::BFloat16& b) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hdiv(a.toCudaBFloat16(), b.toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) / static_cast<float>(b));   
#endif
  return a;
}

CF_HOST_DEVICE
inline coreforge::BFloat16& operator++(coreforge::BFloat16& a) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hadd(a.toCudaBFloat16(), coreforge::BFloat16(1.0f).toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) + 1.0f);  
#endif
  return a;
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator++(coreforge::BFloat16& a, int) {
  coreforge::BFloat16 old(a);
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hadd(a.toCudaBFloat16(), coreforge::BFloat16(1.0f).toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) + 1.0f);
#endif
  return old;
}

CF_HOST_DEVICE
inline coreforge::BFloat16& operator--(coreforge::BFloat16& a) {
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hsub(a.toCudaBFloat16(), coreforge::BFloat16(1.0f).toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) - 1.0f);  
#endif
  return a;
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator--(coreforge::BFloat16& a, int) {
  coreforge::BFloat16 old(a);
#if defined(__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
  a = coreforge::BFloat16(__hsub(a.toCudaBFloat16(), coreforge::BFloat16(1.0f).toCudaBFloat16()));
#else
  a = coreforge::BFloat16(static_cast<float>(a) - 1.0f);  
#endif
  return old;
}
}  // namespace coreforge

// user-defined literals for BFloat16
CF_HOST_DEVICE
inline coreforge::BFloat16 operator"" _bf16(long double value) {
  return coreforge::BFloat16(static_cast<float>(value));
}

CF_HOST_DEVICE
inline coreforge::BFloat16 operator"" _bf16(unsigned long long int value) {
  return coreforge::BFloat16(static_cast<uint32_t>(value));
}
#endif // __COREFORGE_BFLOAT16_H__
