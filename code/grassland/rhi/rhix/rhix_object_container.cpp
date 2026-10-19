#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
ObjectContainer::ObjectContainer() noexcept = default;

ObjectContainer::ObjectContainer(rhi::IObject *object) noexcept : object_(object) {
}

ObjectContainer::ObjectContainer(const ObjectContainer &) noexcept = default;
ObjectContainer::ObjectContainer(ObjectContainer &&) noexcept = default;
ObjectContainer &ObjectContainer::operator=(const ObjectContainer &) noexcept = default;
ObjectContainer &ObjectContainer::operator=(ObjectContainer &&) noexcept = default;
ObjectContainer::~ObjectContainer() = default;

void ObjectContainer::Release() {
  object_.Release();
}

ObjectContainer::operator bool() const noexcept {
  return static_cast<bool>(object_);
}

rhi::IObject *ObjectContainer::GetObject() const noexcept {
  return object_.Get();
}
}  // namespace grassland::rhix
