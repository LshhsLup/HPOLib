#ifndef __HPOLIB_CHECK_H__
#define __HPOLIB_CHECK_H__

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpolib {

#if defined(__GNUC__) || defined(__clang__)
#define HPOLIB_LIKELY(x) __builtin_expect(!!(x), 1)
#define HPOLIB_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HPOLIB_LIKELY(x) (x)
#define HPOLIB_UNLIKELY(x) (x)
#endif

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline void str_helper(std::ostringstream& ss) {}

template <typename T, typename... Args>
inline void str_helper(std::ostringstream& ss, const T& t,
                       const Args&... args) {
  ss << t;
  str_helper(ss, args...);
}

template <typename... Args>
std::string fmt_string(const Args&... args) {
  std::ostringstream ss;
  str_helper(ss, args...);
  return ss.str();
}

#define HPOLIB_CHECK(cond, ...)                                               \
  if (HPOLIB_UNLIKELY(!(cond))) {                                             \
    std::string msg =                                                         \
        hpolib::fmt_string("Check failed: ", #cond, ".\n", __VA_ARGS__, "\n", \
                           "At: ", __FILE__, ":", __LINE__);                  \
    throw hpolib::Error(msg);                                                 \
  }

#define HPOLIB_INTERNAL_ASSERT(cond, ...)                                    \
  if (HPOLIB_UNLIKELY(!(cond))) {                                            \
    std::string msg = hpolib::fmt_string(                                    \
        "Internal Assertion failed: ", #cond, ".\n",                         \
        "Please report this bug to HPOLib developers.\n", __VA_ARGS__, "\n", \
        "At: ", __FILE__, ":", __LINE__);                                    \
    throw hpolib::Error(msg);                                                \
  }

}  // namespace hpolib

#endif  // __HPOLIB_CHECK_H__