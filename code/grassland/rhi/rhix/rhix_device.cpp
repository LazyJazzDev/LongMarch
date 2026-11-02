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

ComputeProgram Device::CreateComputeProgram(const rhi::ComputeProgramDesc &desc) const {
  return Get()->CreateComputeProgram(desc);
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

Sampler Device::CreateSampler(const rhi::SamplerDesc &desc) const {
  return Get()->CreateSampler(desc);
}

BindingGroup Device::CreateBindingGroup(const rhi::BindingLayoutDesc &layout) const {
  return Get()->CreateBindingGroup(layout);
}

BottomLevelAccelerationStructure Device::CreateBottomLevelAccelerationStructure(
    const rhi::BottomLevelAccelerationStructureDesc &desc) const {
  return Get()->CreateBottomLevelAccelerationStructure(desc);
}

TopLevelAccelerationStructure Device::CreateTopLevelAccelerationStructure(
    const rhi::TopLevelAccelerationStructureDesc &desc) const {
  return Get()->CreateTopLevelAccelerationStructure(desc);
}

RayTracingProgram Device::CreateRayTracingProgram(const rhi::RayTracingProgramDesc &desc) const {
  return Get()->CreateRayTracingProgram(desc);
}

Shader Device::CreateShader(const std::string &filename,
                            const std::string &entry_point,
                            rhi::ShaderStage stage,
                            FileSystem *fs,
                            const std::vector<std::string> &include_paths,
                            const std::vector<std::pair<std::string, std::string>> &defines) const {
  return Get()->CreateShader(filename, entry_point, stage, fs, include_paths, defines);
}

Shader Device::CreateShaderFromSource(const std::string &source,
                                      const std::string &entry_point,
                                      rhi::ShaderStage stage,
                                      const std::vector<std::string> &include_paths,
                                      const std::vector<std::pair<std::string, std::string>> &defines) const {
  return Get()->CreateShaderFromSource(source, entry_point, stage, include_paths, defines);
}
}  // namespace grassland::rhix
