#ifndef __COREFORGE_TENSOR_IMPL_H__
#define __COREFORGE_TENSOR_IMPL_H__

#include "DType.h"
#include "options.h"
#include "storage.h"
#include "vectorUtils.h"

namespace coreforge {

constexpr size_t small_vector_inline_size = 5;
using IntArrayView = ArrayView<int64_t>;
using ShapeVector = SmallVector<int64_t, small_vector_inline_size>;
using StrideVector = SmallVector<int64_t, small_vector_inline_size>;

class TensorImpl {
 public:
  TensorImpl(IntArrayView shape, Options options) : options_(options), shape_(shape) {
    ASSERT_MSG(shape.size() <= MAX_TENSOR_DIMS,
               "Tensor dimension exceeds the maximum limit, which is %d",
               MAX_TENSOR_DIMS);
    computeNumel(numel_, shape_);
    computeStrides(strides_, shape_);
    storage_offset_ = 0;
  }

  TensorImpl(IntArrayView shape, Options options,
             const std::shared_ptr<Storage>& storage, int64_t storage_offset) : options_(options), shape_(shape) {
    ASSERT_MSG(shape.size() <= MAX_TENSOR_DIMS,
               "Tensor dimension exceeds the maximum limit, which is %d",
               MAX_TENSOR_DIMS);
    computeNumel(numel_, shape_);
    computeStrides(strides_, shape_);
    ASSERT_MSG(storage_offset * getDTypeSize(options_.dtype_) + nbytes() <= storage->size(),
               "Storage size is smaller than tensor size with given offset");
    storage_offset_ = storage_offset;
    storage_ = storage;
    data_ptr_ = storage->dataPtr<uint8_t>() + storage_offset_ * getDTypeSize(options_.dtype_);
  }

  template <typename T>
  TensorImpl(const std::vector<T>& values, IntArrayView shape,
             Options options = {}) : TensorImpl(shape, options) {
    ASSERT_MSG(static_cast<int64_t>(values.size()) == numel(),
               "Number of values does not match tensor size");
    checkDTypeMatch<T>(options.dtype_);
    Storage::copyOnDevice(
        dataPtr(), options.device_,
        static_cast<const void*>(values.data()), Device::CPU(),
        sizeof(T) * static_cast<int64_t>(values.size()));
  }

  TensorImpl() = default;
  ~TensorImpl() = default;

  TensorImpl(const TensorImpl&) = default;
  TensorImpl& operator=(const TensorImpl&) = default;
  TensorImpl(TensorImpl&&) noexcept = default;
  TensorImpl& operator=(TensorImpl&&) noexcept = default;

  constexpr DType dtype() const { return options_.dtype_; }
  constexpr Device device() const { return options_.device_; }
  constexpr bool pinnedMemory() const { return options_.pinnedMemory_; }
  constexpr bool requiresGrad() const { return options_.requiresGrad_; }

  constexpr int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
  // total number of elements
  constexpr int64_t numel() const { return numel_; }
  constexpr int64_t nbytes() const {
    return numel_ * static_cast<int64_t>(getDTypeSize(options_.dtype_));
  }
  constexpr int64_t storageOffset() const { return storage_offset_; }
  constexpr bool isScalar() const { return shape_.empty(); }

  template <typename T = void>
  T* dataPtr() const {
    ensureStorageAllocated();
    return static_cast<T*>(data_ptr_);
  }

  template <typename T = void>
  const T* dataPtr() const {
    ensureStorageAllocated();
    return static_cast<const T*>(data_ptr_);
  }

  constexpr Options& options() { return options_; }
  constexpr const Options& options() const { return options_; }
  constexpr IntArrayView shape() const { return IntArrayView(shape_); }
  constexpr IntArrayView strides() const { return IntArrayView(strides_); }
  constexpr int64_t shape(int64_t d) const { return shape_[d < 0 ? d + dim() : d]; }
  constexpr int64_t stride(int64_t d) const { return strides_[d < 0 ? d + dim() : d]; }

  const std::shared_ptr<Storage>& storage() const {
    ensureStorageAllocated();
    return storage_;
  }

  void setStorage(const std::shared_ptr<Storage>& storage,
                  int64_t storage_offset = 0) {
    ASSERT_MSG(storage_offset * getDTypeSize(options_.dtype_) + nbytes() <= storage->size(),
               "Storage size is smaller than tensor size with given offset");
    storage_ = storage;
    storage_offset_ = storage_offset;
    data_ptr_ = storage->dataPtr<uint8_t>() + storage_offset_ * getDTypeSize(options_.dtype_);
  }

  void copyOnWrite() const {
    ensureStorageAllocated();
    if (storage_.use_count() > 1) {
      auto new_storage = std::make_shared<Storage>(nbytes(), options_.device_);
      Storage::copyOnDevice(new_storage->dataPtr<uint8_t>(), options_.device_,
                           data_ptr_, options_.device_, nbytes());
      storage_ = new_storage;
      storage_offset_ = 0;
      data_ptr_ = storage_->dataPtr<uint8_t>();
    }
  }

  void reshape_(IntArrayView new_shape) {}

  // flatten from start_dim to end_dim (inclusive)
  void flatten_(int64_t start_dim, int64_t end_dim = -1) {}
  // unflatten at dim with given shape
  void unflatten_(int64_t dim, IntArrayView shape) {}

  void squeeze_(int64_t dim = -1) {}

  void squeeze_(IntArrayView dims) {}

  void unsqueeze_(int64_t dim) {}

  template <typename T>
  std::vector<T> toList() const {}

  template <typename T>
  T item() const {}

 private:
  int64_t numel_{0};
  int64_t storage_offset_{0}; // element offset in the storage
  mutable void* data_ptr_{nullptr};

  Options options_{};
  ShapeVector shape_{};
  StrideVector strides_{};
  mutable std::shared_ptr<Storage> storage_{nullptr};

  void ensureStorageAllocated() const {
    if (!storage_) {
      Allocator* allocator = getAllocator(options_);
      storage_ =
          std::make_shared<Storage>(nbytes(), options_.device_, allocator);
      data_ptr_ = storage_->dataPtr<uint8_t>() + storage_offset_ * getDTypeSize(options_.dtype_);
    }
  }

  void computeNumel(int64_t& numel, const IntArrayView shape) {
    numel = 1;
    for (const auto& dim_size : shape) {
      numel *= dim_size;
    }
  }

  // default stride computation: contiguous layout
  void computeStrides(StrideVector& strides, const IntArrayView shape) {
    strides.resize(shape.size());
    if (shape.empty()) {
      return;
    }
    strides[shape.size() - 1] = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
};

}  // namespace coreforge
#endif  // __COREFORGE_TENSOR_IMPL_H__