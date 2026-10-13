#pragma once
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {

class IObject {
 public:
  void AddRef() noexcept {
    references_.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept {
    if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  virtual ~IObject() = default;

 private:
  std::atomic<uint32_t> references_{1};
};

namespace backend {
// Backend-only typed strong reference. T may be forward-declared at the storage site;
// define the owning class's constructor/destructor where T is complete.
template <typename T>
class ObjectPtr {
 public:
  ObjectPtr() noexcept = default;

  // Consume the initial reference of a newly created object.
  static ObjectPtr Adopt(T *object) noexcept {
    static_assert(std::is_base_of_v<IObject, T>);
    ObjectPtr result;
    result.ptr_ = object;
    return result;
  }

  // The borrowed object must already be kept alive by another owner.
  static ObjectPtr Retain(T *object) noexcept {
    if (object)
      object->AddRef();
    return Adopt(object);
  }

  ObjectPtr(const ObjectPtr &other) noexcept : ObjectPtr(Retain(other.ptr_)) {
  }

  ObjectPtr(ObjectPtr &&other) noexcept : ptr_(other.Detach()) {
  }

  // Only implicit pointer conversions are accepted: derived -> base, never downcasts.
  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  ObjectPtr(const ObjectPtr<U> &other) noexcept : ObjectPtr(Retain(other.Get())) {
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  ObjectPtr(ObjectPtr<U> &&other) noexcept : ptr_(other.Detach()) {
  }

  ObjectPtr &operator=(ObjectPtr other) noexcept {
    Swap(other);
    return *this;
  }

  ~ObjectPtr() {
    if (ptr_)
      ptr_->Release();
  }

  T *Get() const noexcept {
    return ptr_;
  }

  T *operator->() const noexcept {
    return ptr_;
  }

  T &operator*() const noexcept {
    return *ptr_;
  }

  explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  void Reset() noexcept {
    ObjectPtr empty;
    Swap(empty);
  }

  void Swap(ObjectPtr &other) noexcept {
    std::swap(ptr_, other.ptr_);
  }

  // Transfer one owned reference without releasing it; the recipient must adopt it.
  T *Detach() noexcept {
    return std::exchange(ptr_, nullptr);
  }

 private:
  T *ptr_{nullptr};
};

class ICore;
class IDevice;

// The only friend bridge: no public raw-pointer adoption or implementation getter.
struct Access {
  template <typename Impl, typename Wrapper>
  static Impl &As(const Wrapper &wrapper) noexcept {
    return *static_cast<Impl *>(wrapper.object_.obj_);
  }

  template <typename Impl, typename Wrapper>
  static ObjectPtr<Impl> Retain(const Wrapper &wrapper) noexcept {
    return ObjectPtr<Impl>::Retain(static_cast<Impl *>(wrapper.object_.obj_));
  }

  template <typename Wrapper, typename Impl>
  static Wrapper Adopt(std::unique_ptr<Impl> object) noexcept {
    static_assert((std::is_same_v<Wrapper, Core> && std::is_base_of_v<ICore, Impl>) ||
                  (std::is_same_v<Wrapper, Device> && std::is_base_of_v<IDevice, Impl>));
    Wrapper wrapper;
    wrapper.object_.obj_ = object.release();
    return wrapper;
  }
};
}  // namespace backend
}  // namespace grassland::rhi
