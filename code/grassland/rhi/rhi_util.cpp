#include "grassland/rhi/rhi_util.h"

#include <stdexcept>
#include <utility>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/util/log.h"

namespace grassland::rhi {

ObjectRef::ObjectRef(const ObjectRef &other) noexcept : obj_(other.obj_) {
  if (obj_)
    obj_->AddRef();
}

ObjectRef::ObjectRef(ObjectRef &&other) noexcept : obj_(std::exchange(other.obj_, nullptr)) {
}

ObjectRef &ObjectRef::operator=(const ObjectRef &other) noexcept {
  ObjectRef copy(other);
  std::swap(obj_, copy.obj_);
  return *this;
}

ObjectRef &ObjectRef::operator=(ObjectRef &&other) noexcept {
  if (this != &other) {
    ObjectRef moved(std::move(other));
    std::swap(obj_, moved.obj_);
  }
  return *this;
}

ObjectRef::~ObjectRef() {
  if (obj_)
    obj_->Release();
}

ObjectRef::operator bool() const noexcept {
  return obj_ != nullptr;
}

void ReportError(int32_t error_code, const std::string &error_message) {
  LogError("Error {}: {}", error_code, error_message);
}

std::string to_string(BackendAPI api) {
  switch (api) {
    case BACKEND_API_D3D12:
      return "BACKEND_API_D3D12";
    case BACKEND_API_VULKAN:
      return "BACKEND_API_VULKAN";
    case BACKEND_API_METAL:
      return "BACKEND_API_METAL";
    default:
      throw std::runtime_error("Unknown API type");
  }
}

std::string BackendAPIName(BackendAPI api) {
  switch (api) {
    case BACKEND_API_D3D12:
      return "D3D12";
    case BACKEND_API_VULKAN:
      return "Vulkan";
    case BACKEND_API_METAL:
      return "Metal";
    default:
      throw std::runtime_error("Unknown API type");
  }
}

std::string to_string(PhysicalDeviceType type) {
  switch (type) {
    case PHYSICAL_DEVICE_TYPE_INTEGRATED:
      return "PHYSICAL_DEVICE_TYPE_INTEGRATED";
    case PHYSICAL_DEVICE_TYPE_DEDICATED:
      return "PHYSICAL_DEVICE_TYPE_DEDICATED";
    case PHYSICAL_DEVICE_TYPE_SOFTWARE:
      return "PHYSICAL_DEVICE_TYPE_SOFTWARE";
    case PHYSICAL_DEVICE_TYPE_OTHER:
      return "PHYSICAL_DEVICE_TYPE_OTHER";
    default:
      throw std::runtime_error("Unknown PHYSICAL_DEVICE_TYPE");
  }
}

std::string PhysicalDeviceTypeName(PhysicalDeviceType type) {
  switch (type) {
    case PHYSICAL_DEVICE_TYPE_DEDICATED:
      return "Dedicated";
    case PHYSICAL_DEVICE_TYPE_INTEGRATED:
      return "Integrated";
    case PHYSICAL_DEVICE_TYPE_SOFTWARE:
      return "Software";
    case PHYSICAL_DEVICE_TYPE_OTHER:
      return "Other";
    default:
      throw std::runtime_error("Unknown PHYSICAL_DEVICE_TYPE");
  }
}

}  // namespace grassland::rhi
