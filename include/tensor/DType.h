#ifndef __COREFORGE_DTYPE_H__
#define __COREFORGE_DTYPE_H__

#include <cstdint>
#include <string>
#include <type_traits>
#include <typeinfo>
#include "BFloat16.h"
#include "Half.h"
#include "config.h"
#include "utils.h"

namespace coreforge {

enum class DType : int8_t {
  Float32 = 0,
  Float16 = 1,
  BFloat16 = 2,
  Int32 = 3,
  Int64 = 4,
  Bool = 5,
  DTypeCount
};

inline size_t getDTypeSize(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return sizeof(float);
    case DType::Float16:
      return sizeof(Half);
    case DType::BFloat16:
      return sizeof(BFloat16);
    case DType::Int32:
      return sizeof(int32_t);
    case DType::Int64:
      return sizeof(int64_t);
    case DType::Bool:
      return sizeof(bool);
    default:
      return 0;
  }
}

template <DType>
struct DTypeToCPPType {
  using type = void;
};

template <>
struct DTypeToCPPType<DType::Float32> {
  using type = float;
};

template <>
struct DTypeToCPPType<DType::Float16> {
  using type = Half;
};

template <>
struct DTypeToCPPType<DType::BFloat16> {
  using type = BFloat16;
};

template <>
struct DTypeToCPPType<DType::Int32> {
  using type = int32_t;
};

template <>
struct DTypeToCPPType<DType::Int64> {
  using type = int64_t;
};

template <>
struct DTypeToCPPType<DType::Bool> {
  using type = bool;
};

template <typename T>
struct CPPTypeToDType {
  static constexpr DType value = DType::DTypeCount;
};

template <>
struct CPPTypeToDType<float> {
  static constexpr DType value = DType::Float32;
};

template <>
struct CPPTypeToDType<Half> {
  static constexpr DType value = DType::Float16;
};

template <>
struct CPPTypeToDType<BFloat16> {
  static constexpr DType value = DType::BFloat16;
};

template <>
struct CPPTypeToDType<int32_t> {
  static constexpr DType value = DType::Int32;
};

template <>
struct CPPTypeToDType<int64_t> {
  static constexpr DType value = DType::Int64;
};

template <>
struct CPPTypeToDType<bool> {
  static constexpr DType value = DType::Bool;
};

template <DType dtype>
using DTypeToCPPType_t = typename DTypeToCPPType<dtype>::type;

template <typename T>
static inline constexpr DType CPPTypeToDType_v = CPPTypeToDType<T>::value;

template <typename T>
void checkDTypeMatch(DType dtype) {
  if (dtype == DType::Float32) {
    ASSERT_MSG(std::is_same<T, float>::value,
               "DType mismatch: expected float, got %s", typeid(T).name());
  } else if (dtype == DType::Float16) {
    ASSERT_MSG(std::is_same<T, Half>::value,
               "DType mismatch: expected Half, got %s", typeid(T).name());
  } else if (dtype == DType::BFloat16) {
    ASSERT_MSG(std::is_same<T, BFloat16>::value,
               "DType mismatch: expected BFloat16, got %s", typeid(T).name());
  } else if (dtype == DType::Int32) {
    ASSERT_MSG(std::is_same<T, int32_t>::value,
               "DType mismatch: expected int32_t, got %s", typeid(T).name());
  } else if (dtype == DType::Int64) {
    ASSERT_MSG(std::is_same<T, int64_t>::value,
               "DType mismatch: expected int64_t, got %s", typeid(T).name());
  } else if (dtype == DType::Bool) {
    ASSERT_MSG(std::is_same<T, bool>::value,
               "DType mismatch: expected bool, got %s", typeid(T).name());
  } else {
    ASSERT_MSG(false, "Unknown DType");
  }
}

}  // namespace coreforge

#endif