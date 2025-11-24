#ifndef __HPOLIB_SCALAR_H__
#define __HPOLIB_SCALAR_H__

#include <variant>
#include "DType.h"

namespace hpolib {

class Scalar {
 public:
  using ValueType = std::variant<float, Half, BFloat16, int32_t, int64_t, bool>;

  constexpr Scalar(float value) : value_(value) {}
  constexpr Scalar(Half value) : value_(value) {}
  constexpr Scalar(BFloat16 value) : value_(value) {}
  constexpr Scalar(int32_t value) : value_(value) {}
  constexpr Scalar(int64_t value) : value_(value) {}
  constexpr Scalar(bool value) : value_(value) {}

  constexpr DType dtype() const {
    return std::visit(
        [](auto&& arg) -> DType {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, float>)
            return DType::Float32;
          if constexpr (std::is_same_v<T, Half>)
            return DType::Float16;
          if constexpr (std::is_same_v<T, BFloat16>)
            return DType::BFloat16;
          if constexpr (std::is_same_v<T, int32_t>)
            return DType::Int32;
          if constexpr (std::is_same_v<T, int64_t>)
            return DType::Int64;
          if constexpr (std::is_same_v<T, bool>)
            return DType::Bool;
        },
        value_);
  }

  // Convert to type T
  template <typename T>
  constexpr T to() const {
    return std::visit(
        [](auto&& arg) -> T {
          using U = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<U, T>) {
            return arg;
          } else {
            return static_cast<T>(arg);
          }
        },
        value_);
  }

  // convert to DType
  template <DType dtype>
  constexpr DTypeToCPPType_t<dtype> to() const {
    return to<DTypeToCPPType_t<dtype>>();
  }

 private:
  ValueType value_;
};

}  // namespace hpolib

#endif  // __HPOLIB_SCALAR_H__