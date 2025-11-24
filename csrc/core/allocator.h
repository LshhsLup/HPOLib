#ifndef __HPOLIB_ALLOCATOR_H__
#define __HPOLIB_ALLOCATOR_H__

#include <memory>
#include <mutex>
#include <vector>
#include "cuda.h"
#include "device.h"
#include "options.h"

namespace hpolib {

constexpr size_t DEFAULT_ALIGNMENT = 32;

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* allocate(int64_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;
};

template <bool aligned = true>
class CPUAllocator : public Allocator {
 public:
  explicit CPUAllocator(size_t alignment = DEFAULT_ALIGNMENT)
      : alignment_{alignment} {}

  void* allocate(int64_t nbytes) override {
    void* ptr = nullptr;
    if constexpr (aligned) {
#if !defined(_MSC_VER)
      ptr = std::aligned_alloc(alignment_, nbytes);
#else
      ptr = _aligned_malloc(nbytes, alignment_);
#endif
    } else {
      ptr = std::malloc(nbytes);
    }
    return ptr;
  }

  void deallocate(void* ptr) override {
    if constexpr (aligned) {
#if !defined(_MSC_VER)
      std::free(ptr);
#else
      _aligned_free(ptr);
#endif
    } else {
      std::free(ptr);
    }
  }

 private:
  size_t alignment_{DEFAULT_ALIGNMENT};
};

#if defined(__CUDACC__)
using hpolib::cuda::CUDA_CHECK;

class CPUPageLockedAllocator : public Allocator {
 public:
  void* allocate(int64_t nbytes) override {
    void* ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault));
    return ptr;
  }

  void deallocate(void* ptr) override {
    if (ptr) {
      CUDA_CHECK(cudaFreeHost(ptr));
    }
  }
};

class CUDAAllocator : public Allocator {
 public:
  explicit CUDAAllocator(DeviceIndex device_index = 0)
      : device_index_{device_index} {}

  void* allocate(int64_t nbytes) override {
    void* ptr = nullptr;
    cuda::CudaDeviceGuard guard(device_index_);
    CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  void deallocate(void* ptr) override {
    if (ptr) {
      cuda::CudaDeviceGuard guard(device_index_);
      CUDA_CHECK(cudaFree(ptr));
    }
  }

 private:
  DeviceIndex device_index_{0};
};
#endif  // __CUDACC__

Allocator* getAllocator(const Options& options) {
  if (options.device_.isCPU()) {
    if (options.pinnedMemory_) {
#if defined(__CUDACC__)
      static CPUPageLockedAllocator cpu_pinned_allocator;
      return &cpu_pinned_allocator;
#else
      ASSERT_MSG(false, "Pinned memory is not supported without CUDA");
      return nullptr;
#endif
    } else {
      static CPUAllocator<> cpu_allocator;
      return &cpu_allocator;
    }
  } else if (options.device_.isGPU()) {
#if defined(__CUDACC__)
    auto device_count = cuda::getDeviceCount();
    static const std::vector<CUDAAllocator> gpu_allocators = [](int count) {
      std::vector<CUDAAllocator> allocators;
      for (int i = 0; i < count; ++i) {
        allocators.emplace_back(i);
      }
      return allocators;
    }(device_count);
    auto device_index = options.device_.index;
    if (device_index < 0 || device_index >= device_count) {
      LOGE("getAllocator error: Invalid GPU device index %d", device_index);
      return nullptr;
    }
    return &gpu_allocators[device_index];
#else
    ASSERT_MSG(false, "CUDA is not available");
    return nullptr;
#endif
  }
  LOGE("getAllocator error: Unknown device type");
  return nullptr;
}

}  // namespace hpolib
#endif  // __HPOLIB_ALLOCATOR_H__