#include "grassland/rhi/rhi_device.h"

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi {

Device::Device() noexcept = default;

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

using namespace backend;

Window Device::CreateWindow(const WindowDesc &desc) const {
  try {
    if (*this) {
      auto &device = Access::As<IDevice>(*this);
      return Access::Share<Window>(device.ManageWindow(device.CreateWindow(desc)));
    }
    ReportError(-1, "Window creation is unavailable for this device.");
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Window creation ran out of memory.");
  }
  return {};
}

Queue Device::GetQueue(QueueType type) const {
  try {
    return *this ? Access::Adopt<Queue>(Access::As<IDevice>(*this).CreateQueue(type)) : Queue{};
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Queue handle allocation failed.");
    return {};
  }
}

CommandContext Device::CreateCommandContext(const Queue &queue) const {
  if (!*this || !queue)
    return {};
  const auto &impl = Access::As<IQueue>(queue);
  if (impl.Owner() != &Access::As<IDevice>(*this)) {
    ReportError(-1, "The queue belongs to another device.");
    return {};
  }
  try {
    return Access::Adopt<CommandContext>(impl.CreateCommandContext());
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Command context allocation failed.");
    return {};
  }
}

}  // namespace grassland::rhi
