#ifndef __COREFORGE_TENSOR_H__
#define __COREFORGE_TENSOR_H__

#include <memory>
#include <optional>
#include <utility>
#include "scalar.h"
#include "tensorImpl.h"

namespace coreforge {
class Tensor {
 public:
  using TensorPair = std::pair<Tensor, Tensor>;
  constexpr Tensor() = default;
  ~Tensor() = default;
  constexpr Tensor(const Tensor&) = default;
  constexpr Tensor& operator=(const Tensor&) = default;
  constexpr Tensor(Tensor&&) noexcept = default;
  constexpr Tensor& operator=(Tensor&&) noexcept = default;

  bool equals(const Tensor& other) const { return impl_ == other.impl_; }

  Tensor(IntArrayView shape, Options options)
      : impl_(std::make_shared<TensorImpl>(shape, options)) {}

  Tensor(IntArrayView shape, Options options,
         const std::shared_ptr<Storage>& storage, int64_t storage_offset)
      : impl_(std::make_shared<TensorImpl>(shape, options, storage,
                                           storage_offset)) {}

  template <typename T>
  explicit Tensor(const Array1d<T>& values, IntArrayView shape, Options options)
      : impl_(std::make_shared<TensorImpl>(values, shape, options)) {}

  template <typename T>
  explicit Tensor(const Array1d<T>& values, Options options)
      : impl_(std::make_shared<TensorImpl>(
            values, IntArrayView({static_cast<int64_t>(values.size())}),
            options)) {}

  template <typename T>
  explicit Tensor(const Array1d<T>& values, IntArrayView shape)
      : impl_(std::make_shared<TensorImpl>(
            values, shape, options::dtype(CPPTypeToDType_v<T>))) {}

  template <typename T>
  explicit Tensor(const Array1d<T>& values)
      : impl_(std::make_shared<TensorImpl>(
            values, IntArrayView({static_cast<int64_t>(values.size())}),
            options::dtype(CPPTypeToDType_v<T>))) {}

  template <typename T>
  explicit Tensor(const Array2d<T>& values, Options options)
      : impl_(std::make_shared<TensorImpl>(
            flatten2D(values),
            IntArrayView(
                {static_cast<int64_t>(values.size()),
                 static_cast<int64_t>(values.empty() ? 0 : values[0].size())}),
            options)) {}

  template <typename T>
  explicit Tensor(const Array2d<T>& values)
      : impl_(std::make_shared<TensorImpl>(
            flatten2D(values),
            IntArrayView(
                {static_cast<int64_t>(values.size()),
                 static_cast<int64_t>(values.empty() ? 0 : values[0].size())}),
            options::dtype(CPPTypeToDType_v<T>))) {}

  template <typename T>
  explicit Tensor(const Array3d<T>& values, Options options)
      : impl_(std::make_shared<TensorImpl>(
            flatten3D(values),
            IntArrayView(
                {static_cast<int64_t>(values.size()),
                 static_cast<int64_t>(values.empty() ? 0 : values[0].size()),
                 static_cast<int64_t>(values.empty() || values[0].empty()
                                          ? 0
                                          : values[0][0].size())}),
            options)) {}

  template <typename T>
  explicit Tensor(const Array3d<T>& values)
      : impl_(std::make_shared<TensorImpl>(
            flatten3D(values),
            IntArrayView(
                {static_cast<int64_t>(values.size()),
                 static_cast<int64_t>(values.empty() ? 0 : values[0].size()),
                 static_cast<int64_t>(values.empty() || values[0].empty()
                                          ? 0
                                          : values[0][0].size())}),
            options::dtype(CPPTypeToDType_v<T>))) {}

  static Tensor empty(IntArrayView shape, Options options = {}) {
    return Tensor(shape, options);
  }

  static Tensor scalar(const Scalar& value, Options options = {}) {}
  static Tensor ones(IntArrayView shape, Options options = {}) {}
  static Tensor zeros(IntArrayView shape, Options options = {}) {}
  static Tensor full(IntArrayView shape, const Scalar& value,
                     Options options = {}) {}
  static Tensor randn(IntArrayView shape, Options options = {}) {}
  static Tensor rand(IntArrayView shape, Options options = {}) {}
  static Tensor uniform(IntArrayView shape, float min, float max,
                        Options options = {}) {}
  static Tensor bernoulli(IntArrayView shape, float p, Options options = {}) {}

  static Tensor ones_like(const Tensor& other,
                          std::optional<Options> options = std::nullopt) {}
  static Tensor zeros_like(const Tensor& other,
                           std::optional<Options> options = std::nullopt) {}
  static Tensor full_like(const Tensor& other, const Scalar& value,
                          std::optional<Options> options = std::nullopt) {}

  template <typename T>
  static Tensor arrange(T start, T end, T step = static_cast<T>(1),
                        Options options = {}) {}

  template <typename T>
  static Tensor linspace(T start, T end, int64_t steps, Options options = {}) {}

  constexpr bool defined() const { return impl_ != nullptr; }
  constexpr TensorImpl& impl() const { return *impl_; }
  bool pinnedMemory() const { return impl_->pinnedMemory(); }
  bool requiresGrad() const { return impl_->requiresGrad(); }
  // void setRequiresGrad(bool require, const std::shared_ptr<FunctionBase>& fn = nullptr);

  int64_t dim() const { return impl_->dim(); }
  int64_t numel() const { return impl_->numel(); }
  int64_t nbytes() const { return impl_->nbytes(); }
  int64_t storageOffset() const { return impl_->storageOffset(); }
  bool isScalar() const { return impl_->isScalar(); }

  template <typename T = void>
  T* dataPtr() {
    return impl_->dataPtr<T>();
  }

  template <typename T = void>
  const T* dataPtr() const {
    return impl_->dataPtr<T>();
  }

  const Options& options() const { return impl_->options(); }
  IntArrayView shape() const { return impl_->shape(); }
  IntArrayView sizes() const { return impl_->shape(); }
  IntArrayView strides() const { return impl_->strides(); }
  int64_t shape(int64_t d) const { return impl_->shape(d); }
  int64_t size(int64_t d) const { return impl_->shape(d); }
  int64_t stride(int64_t d) const { return impl_->stride(d); }
  const std::shared_ptr<Storage>& storage() const { return impl_->storage(); }

  void copyOnWrite() const { impl_->copyOnWrite(); }
  Tensor clone() const;
  void copy_(const Tensor& src);

  Tensor& grad() const;
  void setGrad(const Tensor& grad) const;
  void setGrad(Tensor&& grad) const;
  void addGrad(const Tensor& grad) const;
  void addGrad(Tensor&& grad) const;
  void zeroGrad() const;

  void backward(const Tensor& grad) const;
  void backward() const;
  bool isLeaf() const;

  template <typename T>
  std::vector<T> toList() const;

  template <typename T>
  T item() const;

  void check() const;

  // convert
  Tensor to(DType type) const;
  Tensor to(Device device) const;
  Tensor to(DType type, Device device) const;

  // fill
  void fill_(const Scalar& val);
  void fillMasked_(const Tensor& mask, const Scalar& val);
  void fillZero_();
  void fillOne_();
  void fillLinSpace_(const Scalar& start, const Scalar& step, int64_t steps);
  void fillUniform_(float min, float max);
  void fillNormal_(float mean = 0.f, float stddev = 1.f);
  void fillBernoulli_(float p);

  // index
  void scatter_(int64_t dim, const Tensor& index, const Tensor& src);
  Tensor tril(int64_t diagonal = 0) const;
  Tensor triu(int64_t diagonal = 0) const;

  // math
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor operator+(const Scalar& other) const;
  Tensor operator-(const Scalar& other) const;
  Tensor operator*(const Scalar& other) const;
  Tensor operator/(const Scalar& other) const;

  friend Tensor operator+(const Scalar& self, const Tensor& other);
  friend Tensor operator-(const Scalar& self, const Tensor& other);
  friend Tensor operator*(const Scalar& self, const Tensor& other);
  friend Tensor operator/(const Scalar& self, const Tensor& other);

  void operator+=(const Tensor& other);
  void operator-=(const Tensor& other);
  void operator*=(const Tensor& other);
  void operator/=(const Tensor& other);

  void operator+=(const Scalar& other);
  void operator-=(const Scalar& other);
  void operator*=(const Scalar& other);
  void operator/=(const Scalar& other);

  Tensor operator<(const Tensor& other) const;
  Tensor operator<=(const Tensor& other) const;
  Tensor operator>(const Tensor& other) const;
  Tensor operator>=(const Tensor& other) const;
  Tensor operator==(const Tensor& other) const;
  Tensor operator!=(const Tensor& other) const;

  Tensor operator<(const Scalar& other) const;
  Tensor operator<=(const Scalar& other) const;
  Tensor operator>(const Scalar& other) const;
  Tensor operator>=(const Scalar& other) const;
  Tensor operator==(const Scalar& other) const;
  Tensor operator!=(const Scalar& other) const;

  Tensor operator~() const;
  Tensor operator&(const Tensor& other) const;
  Tensor operator|(const Tensor& other) const;

  Tensor sin() const;
  Tensor cos() const;
  Tensor sqrt() const;
  Tensor pow(const Scalar& exp) const;
  Tensor pow(const Tensor& exp) const;

  static Tensor maximum(const Tensor& a, const Tensor& b);
  static Tensor minimum(const Tensor& a, const Tensor& b);

  Tensor matmul(const Tensor& other, bool transA = false,
                bool transB = false) const;

  // reduce
  Tensor min() const;
  Tensor max() const;
  Tensor argmin() const;
  Tensor argmax() const;
  Tensor sum() const;
  Tensor mean() const;
  Tensor var(bool unbiased = true) const;
  TensorPair varMean(bool unbiased = true) const;

  TensorPair min(int64_t dim, bool keepDims = false) const;
  TensorPair max(int64_t dim, bool keepDims = false) const;
  Tensor argmin(int64_t dim, bool keepDims = false) const;
  Tensor argmax(int64_t dim, bool keepDims = false) const;
  Tensor sum(int64_t dim, bool keepDims = false) const;
  Tensor mean(int64_t dim, bool keepDims = false) const;
  Tensor var(int64_t dim, bool unbiased, bool keepDims = false) const;
  TensorPair varMean(int64_t dim, bool unbiased, bool keepDims = false) const;

  Tensor sum(IntArrayView dims, bool keepDims = false) const;
  Tensor mean(IntArrayView dims, bool keepDims = false) const;
  Tensor var(IntArrayView dims, bool unbiased = true,
             bool keepDims = false) const;
  TensorPair varMean(IntArrayView dims, bool unbiased = true,
                     bool keepDims = false) const;

  // transform
  void reshape_(IntArrayView shape);
  void permute_(IntArrayView dims);
  void permute_();
  void flatten_(int64_t startDim = 0, int64_t endDim = -1);
  void unflatten_(int64_t dim, IntArrayView shape);
  void squeeze_(int64_t dim = -1);
  void squeeze_(IntArrayView dims);
  void unsqueeze_(int64_t dim);
  void transpose_(int64_t dim0, int64_t dim1);
  void t_();

  Tensor reshape(IntArrayView shape) const;
  Tensor view(IntArrayView shape) const;
  Tensor permute(IntArrayView dims) const;
  Tensor permute() const;
  Tensor flatten(int64_t startDim = 0, int64_t endDim = -1) const;
  Tensor unflatten(int64_t dim, IntArrayView shape) const;
  Tensor squeeze(int64_t dim = -1) const;
  Tensor squeeze(IntArrayView dims) const;
  Tensor unsqueeze(int64_t dim) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor t() const;

  std::vector<Tensor> split(int64_t splitSize, int64_t dim = 0) const;
  std::vector<Tensor> split(IntArrayView sections, int64_t dim = 0) const;
  std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const;

  Tensor narrow(int64_t dim, int64_t start, int64_t length) const;

 private:
  std::shared_ptr<TensorImpl> impl_;
};
}  // namespace coreforge

#endif  // __COREFORGE_TENSOR_H__