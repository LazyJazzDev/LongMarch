#pragma once
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

#include "grassland/rhi/rhi_obj_ptr.h"

namespace grassland::rhi::backend {
class IBackendBindingGroup;
class IBackendBindingVersion;
class IBackendBuffer;
class IBackendCommandContext;
class IBackendComputeProgram;
class IBackendDevice;
class IBackendDownloadToken;
class IBackendImage;
class IBackendProgram;
class IBackendQueue;
class IBackendRayTracingProgram;
class IBackendSampler;
class IBackendShaderBindingTable;
class IBackendSubmissionToken;
class IBackendTimeline;
class IBackendWindow;
}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend {
void ReportError(int32_t error_code, const std::string &error_message);

// Search each registered chain ancestor-first. IObject identity is handled by Object.
template <typename Interface>
void *MatchInterface(InterfaceID id, Interface *source) noexcept {
  if constexpr (std::is_same_v<Interface, IObject>) {
    return nullptr;
  } else {
    using Base = typename InterfaceTraits<Interface>::Base;
    static_assert(!std::is_same_v<Interface, Base> && std::is_convertible_v<Interface *, Base *> &&
                      std::is_base_of_v<IObject, Base>,
                  "Registered parent must be an unambiguous IObject-derived base.");
    if (auto result = MatchInterface<Base>(id, static_cast<Base *>(source)))
      return result;
    return id == uuid<Interface>() ? source : nullptr;
  }
}

// One intrusive counter per native implementation, with no public state layout.
template <typename Primary, typename... Interfaces>
class Object : public Primary, public Interfaces... {
  static_assert(std::is_convertible_v<Primary *, IObject *> && (std::is_convertible_v<Interfaces *, IObject *> && ...),
                "Each interface must have one unambiguous IObject inheritance path.");

 public:
  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

  bool QueryInterface(InterfaceID id, void **output) noexcept final {
    if (!output)
      return false;
    if (id == uuid<IObject>()) {
      *output = static_cast<IObject *>(static_cast<Primary *>(this));
    } else {
      void *result = MatchInterface<Primary>(id, static_cast<Primary *>(this));
      if (!result)
        static_cast<void>(((result = MatchInterface<Interfaces>(id, static_cast<Interfaces *>(this))) || ...));
      *output = result;
    }
    if (!*output)
      return false;
    AddRef();
    return true;
  }

  void AddRef() noexcept final {
    references_.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept final {
    if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }

  bool IsUniquelyOwned() const noexcept {
    return references_.load(std::memory_order_acquire) == 1;
  }

  ~Object() override = default;

 private:
  std::atomic<uint32_t> references_{1};
};

// Backend-only typed strong reference. T may be forward-declared at the storage site;
// define the owning class's constructor/destructor where T is complete.
template <typename T>
class ObjectPtr final {
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

class IBackendDevice;

}  // namespace grassland::rhi::backend
