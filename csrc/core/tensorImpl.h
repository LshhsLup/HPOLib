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
  TensorImpl(IntArrayView shape, Options options)
      : options_(options), shape_(shape) {
    ASSERT_MSG(shape.size() <= MAX_TENSOR_DIMS,
               "Tensor dimension exceeds the maximum limit, which is %d",
               MAX_TENSOR_DIMS);
    computeNumel(numel_, shape_);
    computeStrides(strides_, shape_);
    storage_offset_ = 0;
  }

  TensorImpl(IntArrayView shape, Options options,
             const std::shared_ptr<Storage>& storage, int64_t storage_offset)
      : options_(options), shape_(shape) {
    ASSERT_MSG(shape.size() <= MAX_TENSOR_DIMS,
               "Tensor dimension exceeds the maximum limit, which is %d",
               MAX_TENSOR_DIMS);
    computeNumel(numel_, shape_);
    computeStrides(strides_, shape_);
    ASSERT_MSG(storage_offset * getDTypeSize(options_.dtype_) + nbytes() <=
                   storage->size(),
               "Storage size is smaller than tensor size with given offset");
    storage_offset_ = storage_offset;
    storage_ = storage;
    data_ptr_ = storage->dataPtr<uint8_t>() +
                storage_offset_ * getDTypeSize(options_.dtype_);
  }

  template <typename T>
  TensorImpl(const std::vector<T>& values, IntArrayView shape,
             Options options = {})
      : TensorImpl(shape, options) {
    ASSERT_MSG(static_cast<int64_t>(values.size()) == numel(),
               "Number of values does not match tensor size");
    checkDTypeMatch<T>(options.dtype_);
    Storage::copyOnDevice(
        dataPtr(), options.device_, static_cast<const void*>(values.data()),
        Device::CPU(), sizeof(T) * static_cast<int64_t>(values.size()));
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
  constexpr int64_t shape(int64_t d) const {
    return shape_[d < 0 ? d + dim() : d];
  }
  constexpr int64_t stride(int64_t d) const {
    return strides_[d < 0 ? d + dim() : d];
  }

  const std::shared_ptr<Storage>& storage() const {
    ensureStorageAllocated();
    return storage_;
  }

  void setStorage(const std::shared_ptr<Storage>& storage,
                  int64_t storage_offset = 0) {
    ASSERT_MSG(storage_offset * getDTypeSize(options_.dtype_) + nbytes() <=
                   storage->size(),
               "Storage size is smaller than tensor size with given offset");
    storage_ = storage;
    storage_offset_ = storage_offset;
    data_ptr_ = storage->dataPtr<uint8_t>() +
                storage_offset_ * getDTypeSize(options_.dtype_);
  }

  void copyOnWrite() const {
    // If the tensor is not the only owner of the storage, create a copy
    ensureStorageAllocated();
    if (storage_ && storage_.use_count() > 1) {
      storage_ = storage_->clone();
      data_ptr_ = storage_->dataPtr<uint8_t>() +
                  storage_offset_ * getDTypeSize(options_.dtype_);
    }
  }

  // Replace metadata shape/sizes with `shape`.
  // - supports a single -1 to infer dimension
  // - when shape is empty and numel_ == 1, treat as scalar
  // - validates total numel remains the same
  void reshape_(IntArrayView shape) {
    // set as scalar
    if (shape.empty() && numel_ == 1) {
      shape_.clear();
      strides_.clear();
      return;
    }

    // Prepare new shape container; reserve to avoid reallocations.
    ShapeVector retShape;
    retShape.resize(shape.size());

    int64_t inferredIdx = -1;
    int64_t cnt = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t s = shape[i];
      ASSERT_MSG(s != 0,
                 "reshape_(): dimension must be non-zero (use -1 to infer)");
      if (s == -1) {
        // Only one -1 allowed
        ASSERT_MSG(
            inferredIdx < 0,
            "reshape_(): more than one inferred dimension (-1) specified");
        inferredIdx = static_cast<int64_t>(i);
        retShape[i] = 0;  // placeholder, will fill later
      } else {
        ASSERT_MSG(s > 0, "reshape_(): dimension must be positive or -1");
        cnt *= s;
        retShape[i] = s;
      }
    }

    if (inferredIdx >= 0) {
      ASSERT_MSG(cnt != 0 && (numel_ % cnt) == 0,
                 "reshape_(): cannot infer dimension size");
      retShape[inferredIdx] = numel_ / cnt;
    }

    // verify numel is equal
    int64_t new_numel = 0;
    computeNumel(new_numel, IntArrayView(retShape));
    ASSERT_MSG(new_numel == numel_,
               "reshape_(): number of elements does not match");

    // commit shape and recompute contiguous strides
    shape_ = std::move(retShape);
    computeStrides(strides_, IntArrayView(shape_));
  }

  // flatten from start_dim to end_dim (inclusive).
  // If endDim < 0, treat as last dim.
  void flatten_(int64_t startDim, int64_t endDim = -1) {
    const int64_t D = dim();
    ASSERT_MSG(D >= 0, "flatten_(): invalid tensor dimensionality");
    if (D == 0) {
      // scalar: nothing to flatten
      return;
    }

    // normalize indices
    if (startDim < 0)
      startDim += D;
    ASSERT_MSG(startDim >= 0 && startDim < D,
               "flatten_(): startDim out of range");

    if (endDim < 0)
      endDim += D;
    ASSERT_MSG(endDim >= 0 && endDim < D, "flatten_(): endDim out of range");
    ASSERT_MSG(startDim <= endDim, "flatten_(): startDim must be <= endDim");

    // build new shape
    ShapeVector retShape;
    retShape.reserve(D - (endDim - startDim));
    for (int64_t i = 0; i < startDim; ++i)
      retShape.push_back(shape_[i]);

    int64_t flattenDims = 1;
    for (int64_t i = startDim; i <= endDim; ++i)
      flattenDims *= shape_[i];

    retShape.push_back(flattenDims);

    for (int64_t i = endDim + 1; i < D; ++i)
      retShape.push_back(shape_[i]);

    // reuse reshape_ (validation + stride recompute)
    reshape_(IntArrayView(retShape));
  }

  // unflatten at dim `d` with the given `shape` sub-shape (supports at most one -1 in subshape).
  void unflatten_(int64_t d, IntArrayView shape) {
    const int64_t D = dim();
    ASSERT_MSG(D >= 0, "unflatten_(): invalid tensor dimensionality");
    ASSERT_MSG(D > 0, "unflatten_(): cannot unflatten a scalar");

    if (d < 0)
      d += D;
    ASSERT_MSG(d >= 0 && d < D, "unflatten_(): dim out of range");

    ShapeVector retShape;
    retShape.reserve(D - 1 + shape.size());
    for (int64_t i = 0; i < d; ++i)
      retShape.push_back(shape_[i]);

    int64_t unflattenProd = 1;
    int64_t inferredIdx = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t s = shape[i];
      if (s == -1) {
        ASSERT_MSG(inferredIdx < 0,
                   "unflatten_(): more than one -1 in subshape");
        inferredIdx = static_cast<int64_t>(i);
        retShape.push_back(0);  // placeholder
      } else {
        ASSERT_MSG(s > 0,
                   "unflatten_(): subshape dimensions must be positive or -1");
        unflattenProd *= s;
        retShape.push_back(s);
      }
    }

    if (inferredIdx >= 0) {
      ASSERT_MSG(unflattenProd != 0 && (shape_[d] % unflattenProd) == 0,
                 "unflatten_(): cannot infer sub-dimension");
      retShape[d + inferredIdx] = shape_[d] / unflattenProd;
    } else {
      ASSERT_MSG(unflattenProd == shape_[d],
                 "unflatten_(): product of subshape must match original "
                 "dimension size");
    }

    for (int64_t i = d + 1; i < D; ++i)
      retShape.push_back(shape_[i]);

    // reuse reshape_ for validation and stride recomputation
    reshape_(IntArrayView(retShape));
  }

  // Squeeze a specific dimension `d`. If the dimension is not 1, it's a no-op (PyTorch behavior).
  void squeeze_(int64_t d = -1) {
    if (d == -1) {
      // squeeze all dims equal to 1
      ShapeVector retShape;
      retShape.reserve(shape_.size());
      for (const auto& s : shape_) {
        if (s != 1)
          retShape.push_back(s);
      }
      reshape_(IntArrayView(retShape));
      return;
    }

    const int64_t D = dim();
    ASSERT_MSG(D > 0, "squeeze_(): tensor has no dimensions to squeeze");
    if (d < 0)
      d += D;
    ASSERT_MSG(d >= 0 && d < D, "squeeze_(): dim out of range");

    // If the target dim is not 1, do nothing (compatible with PyTorch).
    if (shape_[d] != 1)
      return;

    ShapeVector retShape;
    retShape.reserve(D - 1);
    for (int64_t i = 0; i < D; ++i) {
      if (i == d)
        continue;
      retShape.push_back(shape_[i]);
    }
    reshape_(IntArrayView(retShape));
  }

  // Squeeze multiple dims: if dims.empty() -> squeeze all size-1 dims.
  // dims can contain negative indices.
  void squeeze_(IntArrayView dims) {
    const int64_t D = dim();
    if (dims.empty()) {
      // squeeze all dims equal to 1
      ShapeVector retShape;
      retShape.reserve(shape_.size());
      for (const auto& s : shape_) {
        if (s != 1)
          retShape.push_back(s);
      }
      reshape_(IntArrayView(retShape));
      return;
    }

    // Build boolean mask for dimensions to remove (validate first)
    std::vector<char> removeMask(static_cast<size_t>(D), 0);
    for (const auto& dd : dims) {
      int64_t idx = dd;
      if (idx < 0)
        idx += D;
      ASSERT_MSG(idx >= 0 && idx < D, "squeeze_(dims): dim out of range");
      // Only mark removal when size == 1 (otherwise behave like PyTorch: ignore)
      if (shape_[idx] == 1)
        removeMask[static_cast<size_t>(idx)] = 1;
    }

    ShapeVector retShape;
    retShape.reserve(D);
    for (int64_t i = 0; i < D; ++i) {
      if (!removeMask[static_cast<size_t>(i)])
        retShape.push_back(shape_[i]);
    }
    reshape_(IntArrayView(retShape));
  }

  // Insert a dimension of size 1 at position d.
  void unsqueeze_(int64_t d) {
    const int64_t D = dim();
    // Valid insertion positions are [0, D] inclusive; support negative indexing
    if (d < 0)
      d += (D + 1);
    ASSERT_MSG(d >= 0 && d <= D,
               "unsqueeze_(): dim out of range for insertion");

    ShapeVector retShape;
    retShape.reserve(D + 1);
    for (int64_t i = 0; i < d; ++i)
      retShape.push_back(shape_[i]);
    retShape.push_back(1);
    for (int64_t i = d; i < D; ++i)
      retShape.push_back(shape_[i]);

    reshape_(IntArrayView(retShape));
  }

  template <typename T>
  std::vector<T> toList() const {
    checkDTypeMatch<T>(dtype());
    if (device().isCPU()) {
      const T* data_ptr = dataPtr<T>();
      return std::vector<T>(data_ptr, data_ptr + numel());
    }
    std::vector<T> host_data(static_cast<size_t>(numel()));
    Storage::copyOnDevice(static_cast<void*>(host_data.data()), Device::CPU(),
                          static_cast<const void*>(dataPtr<T>()), device(),
                          sizeof(T) * numel());
    return host_data;
  }

  template <typename T>
  T item() const {
    ASSERT_MSG(numel() == 1,
               "item() can only be called on tensors with one element");
    checkDTypeMatch<T>(dtype());
    if (device().isCPU()) {
      const T* data_ptr = dataPtr<T>();
      return data_ptr[0];
    }
    T ret;
    Storage::copyOnDevice(static_cast<void*>(&ret), Device::CPU(),
                          static_cast<const void*>(dataPtr<T>()), device(),
                          sizeof(T));
    return ret;
  }

 private:
  int64_t numel_{0};
  int64_t storage_offset_{0};  // element offset in the storage
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
      data_ptr_ = storage_->dataPtr<uint8_t>() +
                  storage_offset_ * getDTypeSize(options_.dtype_);
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