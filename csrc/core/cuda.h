#ifndef __HPOLIB_CUDA_UTILS_H__
#define __HPOLIB_CUDA_UTILS_H__

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "logger.h"
#include "utils.h"

namespace hpolib::cuda {

#if defined(__CUDACC__)

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error in file " << __FILE__ << " at line "           \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

int getDeviceCount() {
  static const int device_count = []() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
  }();
  return device_count;
}

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int device_id) {
    ASSERT_MSG(device_id >= 0 && device_id < getDeviceCount(),
               "Invalid CUDA device ID");
    CUDA_CHECK(cudaGetDevice(&prev_device_));
    if (prev_device_ != device_id) {
      CUDA_CHECK(cudaSetDevice(device_id));
      changed_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (changed_) {
      CUDA_CHECK(cudaSetDevice(prev_device_));
    }
  }

 private:
  int prev_device_{-1};
  bool changed_{false};
};

class CUDAEvent;
class CUDAStream {
 public:
  constexpr CUDAStream() : stream_(nullptr), device_id_(-1) {}
  explicit CUDAStream(int device_id) : stream_(nullptr), device_id_(device_id) {
    CudaDeviceGuard guard(device_id_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }
  CUDAStream(const CUDAStream&) = delete;
  CUDAStream& operator=(const CUDAStream&) = delete;
  CUDAStream(CUDAStream&& other) noexcept
      : stream_(other.stream_), device_id_(other.device_id_) {
    other.stream_ = nullptr;
    other.device_id_ = -1;
  }
  CUDAStream& operator=(CUDAStream&& other) noexcept {
    if (this != &other) {
      destroy();
      stream_ = other.stream_;
      device_id_ = other.device_id_;
      other.stream_ = nullptr;
      other.device_id_ = -1;
    }
    return *this;
  }
  ~CUDAStream() { destroy(); }

  void synchronize() {
    if (stream_) {
      CudaDeviceGuard guard(device_id_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

  void waitStream(const CUDAStream& other) {
    if (!valid() && !other.valid()) {
      LOGE("Both streams are invalid in waitStream");
      return;
    }
    if (device_id_ != other.device_id_) {
      LOGE("Cannot wait on streams from different devices");
      return;
    }
    CudaDeviceGuard guard(device_id_);
    CUDAEvent event(device_id_, cudaEventDisableTiming);
    event.record(other);
    event.block(*this);
  }

  constexpr cudaStream_t stream() const { return stream_; }
  constexpr int device_id() const { return device_id_; }
  constexpr bool valid() const { return stream_ != nullptr; }

 private:
  void destroy() {
    if (stream_) {
      CudaDeviceGuard guard(device_id_);
      CUDA_CHECK(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
  }
  cudaStream_t stream_;
  int device_id_;
};

class CUDAEvent {
 public:
  constexpr CUDAEvent() : event_(nullptr), device_id_(-1) {}
  explicit CUDAEvent(int device_id, unsigned int flags = cudaEventDisableTiming)
      : event_(nullptr), device_id_(device_id) {
    CudaDeviceGuard guard(device_id_);
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
  }

  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  CUDAEvent(CUDAEvent&& other) noexcept
      : event_(other.event_), device_id_(other.device_id_) {
    other.event_ = nullptr;
    other.device_id_ = -1;
  }

  CUDAEvent& operator=(CUDAEvent&& other) noexcept {
    if (this != &other) {
      destroy();
      event_ = other.event_;
      device_id_ = other.device_id_;
      other.event_ = nullptr;
      other.device_id_ = -1;
    }
    return *this;
  }

  ~CUDAEvent() { destroy(); }

  void record(const CUDAStream& stream) const {
    if (!valid() || !stream.valid()) {
      LOGE("Cannot record an invalid event or on an invalid stream");
      return;
    }
    if (device_id_ != stream.device_id()) {
      LOGE("Cannot record event on stream from different device");
      return;
    }
    CudaDeviceGuard guard(device_id_);
    CUDA_CHECK(cudaEventRecord(event_, stream.stream()));
  }

  void block(const CUDAStream& stream) const {
    if (!valid() || !stream.valid()) {
      LOGE("Cannot block on an invalid event or stream");
      return;
    }
    if (device_id_ != stream.device_id()) {
      LOGE("Cannot block event on stream from different device");
      return;
    }
    CudaDeviceGuard guard(device_id_);
    CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), event_));
  }

  bool query() const {
    if (!valid()) {
      LOGE("Cannot query an invalid event");
      return false;
    }
    CudaDeviceGuard guard(device_id_);
    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    } else {
      CUDA_CHECK(err);  // unexpected error
      return false;
    }
  }

  constexpr cudaEvent_t event() const { return event_; }
  constexpr int device_id() const { return device_id_; }
  constexpr bool valid() const { return event_ != nullptr; }

 private:
  void destroy() {
    if (event_) {
      CudaDeviceGuard guard(device_id_);
      CUDA_CHECK(cudaEventDestroy(event_));
      event_ = nullptr;
      device_id_ = -1;
    }
  }
  cudaEvent_t event_;
  int device_id_;
};

static thread_local std::vector<CUDAStream> stream_pool(getDeviceCount());

CUDAStream& getCurrentCUDAStream(int device_id) {
  if (device_id < 0 || device_id >= getDeviceCount()) {
    LOGE("getCurrentCUDAStream error: Invalid CUDA device ID %d", device_id);
    static CUDAStream default_stream{};
    return default_stream;
  }
  auto& stream = stream_pool[device_id];
  if (!stream.valid()) {
    stream = CUDAStream(device_id);
  }
  return stream;
}

#endif  // __CUDACC__

}  // namespace hpolib::cuda

#endif  // __HPOLIB_CUDA_UTILS_H__