#include "grassland/rhi/rhi_device.h"

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi {

std::string Device::Name() const {
  return *this ? backend::Access::As<backend::IDevice>(*this).GetPhysicalDevice().name : std::string{};
}

bool Device::SupportRayTracingPipeline() const {
  return *this && backend::Access::As<backend::IDevice>(*this).GetPhysicalDevice().ray_tracing_pipeline_feature;
}

bool Device::SupportRayQuery() const {
  return *this && backend::Access::As<backend::IDevice>(*this).GetPhysicalDevice().ray_query_feature;
}

bool Device::SupportGeometryShader() const {
  return *this && backend::Access::As<backend::IDevice>(*this).GetPhysicalDevice().geometry_shader_feature;
}

Device::operator bool() const noexcept {
  return static_cast<bool>(object_);
}

}  // namespace grassland::rhi
