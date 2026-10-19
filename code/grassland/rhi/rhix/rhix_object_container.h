#pragma once
#include "grassland/rhi/rhi_obj_ptr.h"

namespace grassland::rhix {
class ObjectContainer {
 public:
  void Release();
  explicit operator bool() const noexcept;

 protected:
  ObjectContainer() noexcept;
  ObjectContainer(rhi::IObject *object) noexcept;
  ObjectContainer(const ObjectContainer &) noexcept;
  ObjectContainer(ObjectContainer &&) noexcept;
  ObjectContainer &operator=(const ObjectContainer &) noexcept;
  ObjectContainer &operator=(ObjectContainer &&) noexcept;
  ~ObjectContainer();
  rhi::IObject *GetObject() const noexcept;

 private:
  rhi::ObjPtr<rhi::IObject> object_;
};
}  // namespace grassland::rhix
