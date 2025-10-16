#ifndef __COREFORGE_DEVICE_H__
#define __COREFORGE_DEVICE_H__

#include <cstdint>
#include <string_view>

namespace coreforge {

enum class DeviceType : int8_t { CPU = 0, GPU = 1, DeviceTypeCount };

using DeviceIndex = int8_t;

static constexpr std::string_view device_type_names[] = {
    "CPU",
    "GPU",
};

inline constexpr std::string_view DeviceTypeToString(DeviceType device_type) {
  return device_type_names[static_cast<size_t>(device_type)];
}

struct Device {
  DeviceType type;
  DeviceIndex index;

  constexpr Device(DeviceType type, DeviceIndex index = 0)
      : type(type), index(index) {}

  constexpr bool operator==(const Device& other) const {
    return type == other.type && index == other.index;
  }

  // is CPU device
  constexpr bool isCPU() const { return type == DeviceType::CPU; }

  // is GPU device
  constexpr bool isGPU() const { return type == DeviceType::GPU; }

  static constexpr Device CPU() { return Device(DeviceType::CPU); }
  static constexpr Device GPU(DeviceIndex index = 0) {
    return Device(DeviceType::GPU, index);
  }
};

}  // namespace coreforge

#endif  // __COREFORGE_DEVICE_H__