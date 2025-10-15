#ifndef __COREFORGE_DTYPE_H__
#define __COREFORGE_DTYPE_H__

#include <cstdint>
#include <type_traits>
#include <typeinfo>
#include <string>
#include "BFloat16.h"
#include "config.h"
#include "Half.h"

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

template<DType>
struct DTypeToCPPType {
  using type = void;
};

template<>
struct DTypeToCPPType<DType::Float32> {
  using type = float;
};

template<>
struct DTypeToCPPType<DType::Float16> {
  using type = Half;
};

template<>
struct DTypeToCPPType<DType::BFloat16> {
  using type = BFloat16;
};

template<>
struct DTypeToCPPType<DType::Int32> {
  using type = int32_t;
};

template<>
struct DTypeToCPPType<DType::Int64> {
  using type = int64_t;
};

template<>
struct DTypeToCPPType<DType::Bool> {
  using type = bool;
};

template<typename T>
struct CPPTypeToDType { 
  static constexpr DType value = DType::DTypeCount;
};

template<>
struct CPPTypeToDType<float> {
  static constexpr DType value = DType::Float32;
};

template<>
struct CPPTypeToDType<Half> {
  static constexpr DType value = DType::Float16;
};

template<>
struct CPPTypeToDType<BFloat16> {
  static constexpr DType value = DType::BFloat16;
};

template<>
struct CPPTypeToDType<int32_t> {
  static constexpr DType value = DType::Int32;
};

template<>
struct CPPTypeToDType<int64_t> {
  static constexpr DType value = DType::Int64;
};

template<>
struct CPPTypeToDType<bool> {
  static constexpr DType value = DType::Bool;
};

template<DType dtype>
using DTypeToCPPType_t = typename DTypeToCPPType<dtype>::type;

template<typename T>
static inline constexpr DType CPPTypeToDType_v = CPPTypeToDType<T>::value;

template<typename T>
void checkDTypeMatch(DType dtype) {
  static_assert(CPPTypeToDType_v<T> != DType::DTypeCount, "Unsupported type");
  std::string debug_info = "DType and C++ type mismatch: expected "
                + std::string(typeid(DTypeToCPPType_t<CPPTypeToDType_v<T>>).name()) + ", got "
                + std::string(typeid(T).name());
  static_assert(std::is_same<T, DTypeToCPPType_t<CPPTypeToDType_v<T>>>::value, debug_info.c_str());
}

} // namespace coreforge

#endif