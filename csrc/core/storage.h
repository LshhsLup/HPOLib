#ifndef __HPOLIB_STORAGE_H__
#define __HPOLIB_STORAGE_H__

#include <functional>
#include <memory>

#include "allocator.h"
#include "device.h"

namespace hpolib {

class Storage {
 public:
  Storage(int64_t nbytes, Device device, Allocator* allocator = nullptr)
      : nbytes_(nbytes),
        device_(device),
        allocator_(allocator ? allocator : getAllocator(Options(device_))),
        data_(nullptr, [](void*) {}) {
    void* ptr = allocator_->allocate(nbytes_);
    data_ = std::unique_ptr<void, std::function<void(void*)>>(
        ptr, [this](void* p) { allocator_->deallocate(p); });
  }

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) noexcept = default;
  Storage& operator=(Storage&&) noexcept = default;

  std::shared_ptr<Storage> clone() const {
    auto new_storage = std::make_shared<Storage>(nbytes_, device_, allocator_);
    copyOnDevice(new_storage->data_.get(), device_, data_.get(), device_,
                 nbytes_);
    return new_storage;
  }

  template <typename T>
  constexpr T* dataPtr() const {
    return static_cast<T*>(data_.get());
  }

  constexpr int64_t size() const { return nbytes_; }
  constexpr Device device() const { return device_; }

  static void copyOnDevice(void* dst, const Device& dst_device, const void* src,
                           const Device& src_device, int64_t nbytes) {
    if (nbytes <= 0) {
      return;
    }
    // cpu -> cpu
    if (dst_device.isCPU() && src_device.isCPU()) {
      std::memcpy(dst, src, nbytes);
      return;
    }
#if defined(__CUDACC__)
    // gpu -> gpu
    if (dst_device.isGPU() && src_device.isGPU()) {
      cuda::CudaDeviceGuard dst_guard(dst_device.index);
      auto& stream = cuda::getCurrentCUDAStream(dst_device.index);
      CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice,
                                 stream.stream()));
      stream.synchronize();
      return;
    }

    // cpu -> gpu
    if (dst_device.isGPU() && src_device.isCPU()) {
      cuda::CudaDeviceGuard dst_guard(dst_device.index);
      auto& stream = cuda::getCurrentCUDAStream(dst_device.index);
      CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice,
                                 stream.stream()));
      stream.synchronize();
      return;
    }

    // gpu -> cpu
    if (dst_device.isCPU() && src_device.isGPU()) {
      cuda::CudaDeviceGuard src_guard(src_device.index);
      auto& stream = cuda::getCurrentCUDAStream(src_device.index);
      CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost,
                                 stream.stream()));
      stream.synchronize();
      return;
    }
#endif
    LOGE("copyOnDevice error: Unsupported device copy from %s to %s",
         DeviceTypeToString(src_device.type).data(),
         DeviceTypeToString(dst_device.type).data());
  }

  static void copyOnDevice(void* dst, const void* src, int64_t nbytes,
                           const Device& device) {
    // copy within the same device
    copyOnDevice(dst, device, src, device, nbytes);
  }

 private:
  int64_t nbytes_;
  Device device_;
  Allocator* allocator_;
  std::unique_ptr<void, std::function<void(void*)>> data_;
};

}  // namespace hpolib

#endif  // __HPOLIB_STORAGE_H__