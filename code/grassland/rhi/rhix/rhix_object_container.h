#pragma once
#include "grassland/rhi/rhi_obj_ptr.h"

namespace grassland::rhix {
class ObjectContainer {
 public:
  // Query the wrapped RHI object; success transfers one owned reference.
  bool QueryInterface(rhi::InterfaceID id, void **output) const noexcept {
    if (object_)
      return object_->QueryInterface(id, output);
    if (output)
      *output = nullptr;
    return false;
  }

  template <typename Interface>
  Interface *QueryInterface() const noexcept {
    return object_ ? object_->QueryInterface<Interface>() : nullptr;
  }

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
  // Preserves the interface subobject passed by the typed wrapper constructor;
  // this is not a QueryInterface(IObject) canonical identity conversion.
  // Typed wrappers can recover that same interface without querying it again.
  rhi::IObject *GetObject() const noexcept;

 private:
  rhi::ObjPtr<rhi::IObject> object_;
};
}  // namespace grassland::rhix
