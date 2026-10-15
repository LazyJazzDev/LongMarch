#include "grassland/rhi/rhix/rhix_device.h"

namespace grassland::rhix {
Device::Device() noexcept = default;

Device::Device(rhi::IDevice *object) noexcept : ObjectContainer(object) {
}

rhi::IDevice *Device::Get() const noexcept {
  return static_cast<rhi::IDevice *>(GetObject());
}

std::string Device::Name() const {
  return *this ? Get()->Name() : std::string{};
}

bool Device::SupportRayTracingPipeline() const {
  return *this && Get()->SupportRayTracingPipeline();
}

bool Device::SupportRayQuery() const {
  return *this && Get()->SupportRayQuery();
}

bool Device::SupportGeometryShader() const {
  return *this && Get()->SupportGeometryShader();
}

Window Device::CreateGLFWWindow(const rhi::WindowDesc &desc) const {
  return *this ? Get()->CreateGLFWWindow(desc) : nullptr;
}

Queue Device::GetQueue(rhi::QueueType type) const {
  return *this ? Get()->GetQueue(type) : nullptr;
}

CommandContext Device::CreateCommandContext() const {
  return *this ? Get()->CreateCommandContext() : nullptr;
}
}  // namespace grassland::rhix
