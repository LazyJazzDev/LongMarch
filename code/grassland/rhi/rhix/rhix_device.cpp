#include "grassland/rhi/rhix/rhix_device.h"

namespace grassland::rhix {
Device::Device() noexcept = default;

Device::Device(rhi::IDevice *object) noexcept : ObjectContainer(object) {
}

rhi::IDevice *Device::Get() const noexcept {
  return static_cast<rhi::IDevice *>(GetObject());
}

std::string Device::Name() const {
#if !defined(NDEBUG)
  if (!*this)
    return std::string{};
#endif
  return Get()->Name();
}

bool Device::SupportRayTracingPipeline() const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SupportRayTracingPipeline();
}

bool Device::SupportRayQuery() const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SupportRayQuery();
}

bool Device::SupportGeometryShader() const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SupportGeometryShader();
}

Window Device::CreateGLFWWindow(const rhi::WindowDesc &desc) const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->CreateGLFWWindow(desc);
}

Queue Device::GetQueue(rhi::QueueType type) const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->GetQueue(type);
}

CommandContext Device::CreateCommandContext() const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->CreateCommandContext();
}

Shader Device::CreateShader(const rhi::ShaderDesc &desc) const {
  return Get()->CreateShader(desc);
}

Program Device::CreateProgram(const rhi::ProgramDesc &desc) const {
  return Get()->CreateProgram(desc);
}

Image Device::CreateImage(const rhi::ImageDesc &desc) const {
  return Get()->CreateImage(desc);
}

Buffer Device::CreateBuffer(const rhi::BufferDesc &desc) const {
  return Get()->CreateBuffer(desc);
}

}  // namespace grassland::rhix
