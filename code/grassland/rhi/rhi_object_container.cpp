#include "grassland/rhi/rhi_object_container.h"

namespace grassland::rhi {

ObjectContainer::ObjectContainer() noexcept = default;
ObjectContainer::ObjectContainer(const ObjectContainer &) noexcept = default;
ObjectContainer::ObjectContainer(ObjectContainer &&) noexcept = default;
ObjectContainer &ObjectContainer::operator=(const ObjectContainer &) noexcept = default;
ObjectContainer &ObjectContainer::operator=(ObjectContainer &&) noexcept = default;
ObjectContainer::~ObjectContainer() = default;

void ObjectContainer::Release() {
  object_ = {};
}

ObjectContainer::operator bool() const noexcept {
  return static_cast<bool>(object_);
}

}  // namespace grassland::rhi
