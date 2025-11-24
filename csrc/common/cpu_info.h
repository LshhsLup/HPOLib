#pragma once

#include <array>
#include <string_view>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cstring>

namespace hpolib {
namespace cpu {
class CPUInfo {
 public:
  static CPUInfo& get_instance() {
    static CPUInfo instance;
    return instance;
  }

  constexpr bool supports_avx512f() const { return supports_avx512f_; }
  constexpr bool supports_avx512dq() const { return supports_avx512dq_; }
  constexpr bool supports_avx512vl() const { return supports_avx512vl_; }
  constexpr bool supports_avx512() const {
    return supports_avx512f_ && supports_avx512dq_ && supports_avx512vl_;
  }

  constexpr const std::string_view& vendor() const {
    return std::string_view(vendor_.data());
  }
  constexpr const std::string_view& brand() const {
    return std::string_view(brand_.data());
  }

 private:
  CPUInfo() { detect_features(); }

  void detect_features() {
    unsigned int regs[4] = {0};
#ifdef _MSC_VER
    __cpuid((int*)regs, 0);
#else
    __cpuid(0, regs[0], regs[1], regs[2], regs[3]);
#endif
    memcpy(vendor_.data(), &regs[1], 4);
    memcpy(vendor_.data() + 4, &regs[3], 4);
    memcpy(vendor_.data() + 8, &regs[2], 4);

#ifdef _MSC_VER
    __cpuid((int*)regs, 1);
#else
    __cpuid(1, regs[0], regs[1], regs[2], regs[3]);
#endif

#ifdef _MSC_VER
    __cpuidex((int*)regs, 7, 0);
#else
    __cpuid_count(7, 0, regs[0], regs[1], regs[2], regs[3]);
#endif

    supports_avx512f_ = regs[1] & (1 << 16);   // EBX bit 16
    supports_avx512dq_ = regs[1] & (1 << 17);  // EBX bit 17
    supports_avx512vl_ = regs[1] & (1 << 31);  // EBX bit 31

#ifdef _MSC_VER
    __cpuid((int*)regs, 0x80000002);
    memcpy(brand_, regs, sizeof(regs));
    __cpuid((int*)regs, 0x80000003);
    memcpy(brand_ + 16, regs, sizeof(regs));
    __cpuid((int*)regs, 0x80000004);
    memcpy(brand_ + 32, regs, sizeof(regs));
#else
    __cpuid(0x80000002, regs[0], regs[1], regs[2], regs[3]);
    memcpy(brand_.data(), regs, sizeof(regs));
    __cpuid(0x80000003, regs[0], regs[1], regs[2], regs[3]);
    memcpy(brand_.data() + 16, regs, sizeof(regs));
    __cpuid(0x80000004, regs[0], regs[1], regs[2], regs[3]);
    memcpy(brand_.data() + 32, regs, sizeof(regs));
#endif
  }

  bool supports_avx512f_ = false;
  bool supports_avx512dq_ = false;
  bool supports_avx512vl_ = false;
  // vendor and brand stored in fixed buffers (no heap alloc)
  std::array<char, 13> vendor_{};  // 12 chars + null
  std::array<char, 49> brand_{};   // 48 chars + null
};

}  // namespace cpu

}  // namespace hpolib