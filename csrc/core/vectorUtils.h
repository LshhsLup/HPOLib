#ifndef __HPOLIB_VECTOR_UTILS_H__
#define __HPOLIB_VECTOR_UTILS_H__

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>
#include "utils.h"

namespace hpolib {
template <typename T>
class ArrayView;

// A small vector class that optimizes for small sizes by storing elements
template <typename T, size_t N = 5>
class SmallVector {
 public:
  using size_type = size_t;

  constexpr SmallVector() : size_(0), capacity_(N), data_(get_inline_data()) {}

  explicit SmallVector(size_type count) : SmallVector() {
    reserve(count);
    std::uninitialized_value_construct_n(data_, count);
    size_ = count;
  }

  SmallVector(size_type count, const T& value) : SmallVector() {
    reserve(count);
    std::uninitialized_fill_n(data_, count, value);
    size_ = count;
  }

  SmallVector(std::initializer_list<T> init) : SmallVector() {
    init_from_range(init);
  }

  SmallVector(const std::vector<T>& vec) : SmallVector() {
    static_assert(
        !std::is_same_v<T, bool>,
        "SmallVector<bool> can NOT be constructed from std::vector<bool>");
    init_from_range(vec);
  }

  SmallVector(const ArrayView& view) : SmallVector() { init_from_range(view); }

  SmallVector(const SmallVector& other) : SmallVector() {
    init_from_range(other);
  }

  SmallVector(SmallVector&& other) noexcept
      : size_(other.size_), capacity_(other.capacity_) {
    if (other.data_ == other.get_inline_data()) {
      data_ = get_inline_data();
      std::uninitialized_move(other.data_, other.data_ + other.size_, data_);
      other.clear();
    } else {
      data_ = other.data_;
      other.data_ = other.get_inline_data();
      other.capacity_ = N;
      other.size_ = 0;
    }
  }

  SmallVector& operator=(const SmallVector& other) {
    if (this != &other) {
      clear();
      init_from_range(other);
    }
    return *this;
  }

  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this != &other) {
      clear();
      if (data_ != get_inline_data()) {
        ::operator delete[](data_, static_cast<std::align_val_t>(alignof(T)));
      }
      size_ = other.size_;
      capacity_ = other.capacity_;
      if (other.data_ == other.get_inline_data()) {
        data_ = get_inline_data();
        std::uninitialized_move(other.data_, other.data_ + other.size_, data_);
        other.clear();
      } else {
        data_ = other.data_;
        other.data_ = other.get_inline_data();
        other.capacity_ = N;
        other.size_ = 0;
      }
    }
    return *this;
  }

  constexpr bool operator==(const ArrayView& other) const {
    return other == *this;
  }
  constexpr bool operator!=(const ArrayView& other) const {
    return other != *this;
  }

  ~SmallVector() {
    clear();
    if (data_ != get_inline_data()) {
      ::operator delete[](data_, static_cast<std::align_val_t>(alignof(T)));
    }
  }

  template <typename... Args>
  void push_back(Args&&... args) {
    if (size_ == capacity_) {
      increase_capacity();
    }
    new (data_ + size_) T(std::forward<Args>(args)...);
    ++size_;
  }

  void resize(size_type new_size) {
    if (new_size > size_) {
      reserve(new_size);
      std::uninitialized_value_construct_n(data_ + size_, new_size - size_);
    } else if (new_size < size_) {
      std::destroy(data_ + new_size, data_ + size_);
    }
    size_ = new_size;
  }

  void resize(size_type new_size, const T& value) {
    if (new_size > size_) {
      reserve(new_size);
      std::uninitialized_fill_n(data_ + size_, new_size - size_, value);
    } else if (new_size < size_) {
      std::destroy(data_ + new_size, data_ + size_);
    }
    size_ = new_size;
  }

  T* insert(T* pos, const T& value) {
    size_type index = pos - data_;
    if (size_ == capacity_) {
      increase_capacity();
    }
    if (index < size_) {
      new (data_ + size_) T(std::move(data_[size_ - 1]));
      for (size_type i = size_ - 1; i > index; --i) {
        data_[i] = std::move(data_[i - 1]);
      }
      data_[index] = value;
    } else {
      new (data_ + size_) T(value);
    }
    ++size_;
    return data_ + index;
  }

  void reserve(size_type new_cap) {
    if (new_cap > capacity_) {
      reallocate(new_cap);
    }
  }

  void clear() {
    std::destroy(data_, data_ + size_);
    size_ = 0;
  }

  constexpr bool empty() const { return size_ == 0; }
  constexpr size_type size() const { return size_; }
  constexpr size_type capacity() const { return capacity_; }
  constexpr T* data() { return data_; }
  constexpr const T* data() const { return data_; }
  constexpr T& operator[](size_type index) { return data_[index]; }
  constexpr const T& operator[](size_type index) const { return data_[index]; }
  constexpr T* begin() { return data_; }
  constexpr T* end() { return data_ + size_; }
  constexpr const T* begin() const { return data_; }
  constexpr const T* end() const { return data_ + size_; }
  constexpr T& front() { return data_[0]; }
  constexpr const T& front() const { return data_[0]; }
  constexpr T& back() { return data_[size_ - 1]; }
  constexpr const T& back() const { return data_[size_ - 1]; }
  constexpr ArrayView<T> view() const { return ArrayView<T>(data_, size_); }

 private:
  constexpr T* get_inline_data() { return reinterpret_cast<T*>(inline_data_); }

  constexpr const T* get_inline_data() const {
    return reinterpret_cast<const T*>(inline_data_);
  }

  void increase_capacity() {
    size_type new_cap = capacity_ * 2;
    reallocate(new_cap);
  }

  void reallocate(size_type new_cap) {
    T* new_data = static_cast<T*>(::operator new[](
        new_cap * sizeof(T), static_cast<std::align_val_t>(alignof(T))));
    std::uninitialized_move(data_, data_ + size_, new_data);
    std::destroy(data_, data_ + size_);
    if (data_ != get_inline_data()) {
      ::operator delete[](data_, static_cast<std::align_val_t>(alignof(T)));
    }
    data_ = new_data;
    capacity_ = new_cap;
  }

  template <typename Range>
  void init_from_range(const Range& range) {
    reserve(range.size());
    std::uninitialized_copy(range.begin(), range.end(), data_);
    size_ = range.size();
  }

  size_type size_;
  size_type capacity_;
  // pointer to current data
  // if size_ <= N, data_ points to inline_data_
  // else data_ points to heap-allocated memory
  T* data_;
  alignas(T) char inline_data_[N * sizeof(T)];  // stack storage for small sizes
};

template <typename T>
class ArrayView {
 public:
  using size_type = size_t;
  using const_iterator = const T*;
  using value_type = T;

  constexpr ArrayView() : data_(nullptr), size_(0) {}

  constexpr ArrayView(const T* data, size_type size)
      : data_(data), size_(size) {}

  ArrayView(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {
    std::static_assert(
        !std::is_same_v<T, bool>,
        "ArrayView<bool> can NOT be constructed from std::vector<bool>");
  }

  ArrayView(const SmallVector<T>& vec) : data_(vec.data()), size_(vec.size()) {}

  constexpr ArrayView(const std::initializer_list<T>& init)
      : data_(init.begin() == init.end() ? static_cast<T*>(nullptr)
                                         : std::begin(init)),
        size_(init.size()) {}

  ArrayView(const ArrayView& other) = default;
  ArrayView& operator=(const ArrayView& other) = default;

  constexpr const T& operator[](size_type index) const {
    ASSERT_MSG(index < size_, "ArrayView index out of bound");
    return data_[index];
  }

  constexpr const T& front() const {
    ASSERT_MSG(size_ > 0, "ArrayView is empty");
    return data_[0];
  }

  constexpr const T& back() const {
    ASSERT_MSG(size_ > 0, "ArrayView is empty");
    return data_[size_ - 1];
  }

  constexpr bool operator==(const ArrayView& other) const {
    if (size_ != other.size_) {
      return false;
    }
    for (size_type i = 0; i < size_; ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }

  constexpr bool operator!=(const ArrayView& other) const {
    return !(*this == other);
  }

  constexpr size_type size() const { return size_; }
  constexpr bool empty() const { return size_ == 0; }
  constexpr const T* data() const { return data_; }

  constexpr const_iterator begin() const { return data_; }
  constexpr const_iterator end() const { return data_ + size_; }

 private:
  const T* data_;
  size_t size_;
};

}  // namespace hpolib

#endif  // __HPOLIB_VECTOR_UTILS_H__