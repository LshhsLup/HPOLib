#ifndef __HPOLIB_UTILS_H__
#define __HPOLIB_UTILS_H__

#include <cstdlib>
#include "logger.h"

namespace hpolib {

// assert
#ifdef _MSC_VER
#include <intrin.h>
#define DEBUG_BREAK() __debugbreak()
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) || defined(__x86_64__)
#define DEBUG_BREAK() __asm__ volatile("int3")
#else
#define DEBUG_BREAK() __builtin_trap()
#endif
#else
#include <cstdlib>
#define DEBUG_BREAK() std::abort()
#endif

// ASSERT: simple form
#define ASSERT(expr)                         \
  do {                                       \
    if (!(expr)) {                           \
      LOGE("Assertion failed: (%s)", #expr); \
      DEBUG_BREAK();                         \
      std::abort();                          \
    }                                        \
  } while (0)

// ASSERT_MSG: printf-style extra message
// Usage: ASSERT_MSG(x != 0, "x must not be zero, got %d", x);
#define ASSERT_MSG(expr, fmt, ...)                                  \
  do {                                                              \
    if (!(expr)) {                                                  \
      LOGE("Assertion failed: (%s) -- " fmt, #expr, ##__VA_ARGS__); \
      DEBUG_BREAK();                                                \
      std::abort();                                                 \
    }                                                               \
  } while (0)

// align
#ifdef _MSC_VER
#define ALIGN(n) __declspec(align(n))
#else
#define ALIGN(n) __attribute__((aligned(n)))
#endif

// align to 16 bytes
#define ALIGN16 ALIGN(16)
}  // namespace hpolib

#endif  // __CFORGE_UTILS_H__