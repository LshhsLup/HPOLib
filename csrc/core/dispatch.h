#ifndef __HPOLIB_DISPATCH_H__
#define __HPOLIB_DISPATCH_H__

#include <iostream>
#include <string>

#include "DType.h"
#include "device.h"
#include "options.h"
#include "tensor.h"

namespace hpolib {

struct DispatchKey {
  DeviceType device;
  DType dtype;
  size_t index() const {
    return static_cast<size_t>(device) *
               static_cast<size_t>(DType::DTypeCount) +
           static_cast<size_t>(dtype);
  }
};

template <typename... Args>
DispatchKey get_key(const Args&... args) {
  // find the first tensor and use its device/dtype to
  // decide the kernel implement.
  const Tensor* t = nullptr;
  auto finder = [&](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, Tensor>) {
      if (!t)
        t = &arg;
    }
  };
  (finder(args), ...);  // Fold expression

  if (t)
    return {t->device(), t->dtype()};
  return {Device::CPU, DType::Float32};  // default fallback
}

template <typename Tag, typename Fn>
struct OpRegistry {
  static constexpr size_t N = static_cast<size_t>(DeviceType::DeviceTypeCount) *
                              static_cast<size_t>(DType::DTypeCount);
  static Fn table[N];

  static void register_impl(DispatchKey key, Fn fn) { table[key.index()] = fn; }
  static Fn lookup(DispatchKey key) { return table[key.index()]; }
};

// initialize static array
template <typename Tag, typename Fn>
Fn OpRegistry<Tag, Fn>::table[OpRegistry<Tag, Fn>::N] = {nullptr};

#define DEFINE_DISPATCH(OP_NAME, SIG)                                      \
  struct OP_NAME##Tag {};                                                  \
  using OP_NAME##Fn = SIG;                                                 \
  using OP_NAME##Registry = hpolib::OpRegistry<OP_NAME##Tag, OP_NAME##Fn>; \
  template <typename... Args>                                              \
  inline auto OP_NAME##_dispatch(Args&&... args) {                         \
    auto key = hpolib::get_key(args...);                                   \
    auto fn = OP_NAME##Registry::lookup(key);                              \
    if (!fn) {                                                             \
      std::cerr << "Kernel not found for " << #OP_NAME << std::endl;       \
      exit(1);                                                             \
    }                                                                      \
    return fn(std::forward<Args>(args)...);                                \
  }

#define REGISTER_IMPL(OP_NAME, DEVICE, DTYPE, FUNC)            \
  static struct Reg##OP_NAME##DEVICE##DTYPE {                  \
    Reg##OP_NAME##DEVICE##DTYPE() {                            \
      OP_NAME##Registry::register_impl({DEVICE, DTYPE}, FUNC); \
    }                                                          \
  } reg_##OP_NAME##DEVICE##DTYPE##_;

}  // namespace hpolib

#endif  // __HPOLIB_DISPATCH_H__