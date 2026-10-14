#pragma once
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {

// Release all caller-owned children before their parent. Releasing a handle
// drops only this reference; submitted work retains its resources until done.
class ObjectContainer {
 public:
  void Release();
  explicit operator bool() const noexcept;

 protected:
  ObjectContainer() noexcept;
  ObjectContainer(const ObjectContainer &) noexcept;
  ObjectContainer(ObjectContainer &&) noexcept;
  ObjectContainer &operator=(const ObjectContainer &) noexcept;
  ObjectContainer &operator=(ObjectContainer &&) noexcept;
  ~ObjectContainer();

 private:
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
