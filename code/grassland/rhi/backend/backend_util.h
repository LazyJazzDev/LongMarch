#pragma once
#include <atomic>
#include <memory>
#include <type_traits>

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
class ICore;
class IDevice;

// The only friend bridge: no public raw-pointer adoption or implementation getter.
struct Access {
  template <typename Impl, typename Wrapper>
  static Impl &As(const Wrapper &wrapper) noexcept {
    return *static_cast<Impl *>(wrapper.object_.obj_);
  }

  template <typename Wrapper>
  static const ObjectRef &Ref(const Wrapper &wrapper) noexcept {
    return wrapper.object_;
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
