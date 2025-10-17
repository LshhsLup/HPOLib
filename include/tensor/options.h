#ifndef __COREFORGE_OPTIONS_H__
#define __COREFORGE_OPTIONS_H__

#include "DType.h"
#include "device.h"

namespace coreforge {

namespace options {
// Thread-local default options for tensor creation.
struct DefaultOptions {
  static Device& device() {
    thread_local static Device default_device = Device::CPU();
    return default_device;
  }

  static DType& dtype() {
    thread_local static DType default_dtype = DType::Float32;
    return default_dtype;
  }
};
}  // namespace options

[[maybe_unused]] static void setDefaultDevice(Device device) {
  options::DefaultOptions::device() = device;
}

[[maybe_unused]] static void setDefaultDType(DType dtype) {
  options::DefaultOptions::dtype() = dtype;
}

// A struct for configuring tensor options
struct Options {
  Device device_;
  DType dtype_;
  bool requiresGrad_;
  bool pinnedMemory_;

  // Intentionally not explicit to allow for convenient conversions, mimicking PyTorch's API.
  constexpr Options(Device device = options::DefaultOptions::device(),
                    DType dtype = options::DefaultOptions::dtype(),
                    bool requiresGrad = false, bool pinnedMemory = false)
      : device_(device),
        dtype_(dtype),
        requiresGrad_(requiresGrad),
        pinnedMemory_(pinnedMemory) {}

  constexpr Options& device(const Device& d) {
    device_ = d;
    return *this;
  }

  constexpr Options& dtype(DType t) {
    dtype_ = t;
    return *this;
  }

  constexpr Options& requiresGrad(bool rg = true) {
    requiresGrad_ = rg;
    return *this;
  }

  constexpr Options& pinnedMemory(bool pm = true) {
    pinnedMemory_ = pm;
    return *this;
  }

  [[nodiscard]] constexpr Options noGrad() const {
    Options ret = *this;
    ret.requiresGrad_ = false;
    return ret;
  }

  [[nodiscard]] constexpr Options indices() const {
    Options ret = *this;
    ret.dtype_ = DType::Int64;
    return ret;
  }
};

namespace options {

constexpr inline Options device(DeviceType type, DeviceIndex index = 0) {
  return Options().device({type, index});
}

constexpr inline Options device(const Device& d) {
  return Options().device(d);
}

constexpr inline Options dtype(DType dt) {
  return Options().dtype(dt);
}

constexpr inline Options requiresGrad(bool rg = true) {
  return Options().requiresGrad(rg);
}

constexpr inline Options pinnedMemory(bool pm = true) {
  return Options().pinnedMemory(pm);
}

}  // namespace options

}  // namespace coreforge

#endif