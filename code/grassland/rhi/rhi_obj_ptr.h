#pragma once
#include <type_traits>
#include <utility>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {

// A raw pointer transfers one owned reference. A container lvalue shares a
// reference; a container rvalue transfers it. Get() returns only a borrowed pointer.
template <typename T>
class ObjPtr {
 public:
  ObjPtr() noexcept = default;

  ObjPtr(T *object) noexcept : object_(object) {
  }

  static ObjPtr Retain(T *object) noexcept {
    if (object)
      object->AddRef();
    return object;
  }

  ObjPtr(const ObjPtr &other) noexcept : ObjPtr(Retain(other.object_)) {
  }

  ObjPtr(ObjPtr &&other) noexcept : object_(other.Detach()) {
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  ObjPtr(const ObjPtr<U> &other) noexcept : ObjPtr(Retain(other.Get())) {
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  ObjPtr(ObjPtr<U> &&other) noexcept : object_(other.Detach()) {
  }

  ObjPtr &operator=(ObjPtr other) noexcept {
    std::swap(object_, other.object_);
    return *this;
  }

  ~ObjPtr() {
    Release();
  }

  void Release() noexcept {
    auto object = std::exchange(object_, nullptr);
    if (object)
      object->Release();
  }

  T *Get() const noexcept {
    return object_;
  }

  T *Detach() noexcept {
    return std::exchange(object_, nullptr);
  }

  T *operator->() const noexcept {
    return object_;
  }

  T &operator*() const noexcept {
    return *object_;
  }

  explicit operator bool() const noexcept {
    return object_ != nullptr;
  }

 private:
  T *object_{};
};

}  // namespace grassland::rhi
