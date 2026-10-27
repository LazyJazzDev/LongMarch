#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhix/rhix_acceleration_structure.h"
#include "grassland/rhi/rhix/rhix_binding_group.h"
#include "grassland/rhi/rhix/rhix_buffer.h"
#include "grassland/rhi/rhix/rhix_command_context.h"
#include "grassland/rhi/rhix/rhix_compute_program.h"
#include "grassland/rhi/rhix/rhix_object_container.h"
#include "grassland/rhi/rhix/rhix_program.h"
#include "grassland/rhi/rhix/rhix_queue.h"
#include "grassland/rhi/rhix/rhix_ray_tracing_program.h"
#include "grassland/rhi/rhix/rhix_shader.h"
#include "grassland/rhi/rhix/rhix_window.h"

namespace grassland::rhix {

class Device : public ObjectContainer {
 public:
  Device(rhi::IDevice *object) noexcept;
  rhi::IDevice *Get() const noexcept;
  Device() noexcept;
  std::string Name() const;
  bool SupportRayTracingPipeline() const;
  bool SupportRayQuery() const;
  bool SupportGeometryShader() const;
  Sampler CreateSampler(const rhi::SamplerDesc &desc = {}) const;
  BindingGroup CreateBindingGroup(const rhi::BindingLayoutDesc &layout) const;
  BottomLevelAccelerationStructure CreateBottomLevelAccelerationStructure(
      const rhi::BottomLevelAccelerationStructureDesc &desc) const;
  TopLevelAccelerationStructure CreateTopLevelAccelerationStructure(
      const rhi::TopLevelAccelerationStructureDesc &desc) const;
  RayTracingProgram CreateRayTracingProgram(const rhi::RayTracingProgramDesc &desc) const;
  Shader CreateShader(const rhi::ShaderDesc &desc) const;
  Shader CreateShader(const std::string &filename,
                      const std::string &entry_point,
                      rhi::ShaderStage stage,
                      FileSystem *fs = nullptr,
                      const std::vector<std::string> &include_paths = {},
                      const std::vector<std::pair<std::string, std::string>> &defines = {}) const;
  Shader CreateShaderFromSource(const std::string &source,
                                const std::string &entry_point,
                                rhi::ShaderStage stage,
                                const std::vector<std::string> &include_paths = {},
                                const std::vector<std::pair<std::string, std::string>> &defines = {}) const;
  ComputeProgram CreateComputeProgram(const rhi::ComputeProgramDesc &desc) const;
  Program CreateProgram(const rhi::ProgramDesc &desc) const;
  Image CreateImage(const rhi::ImageDesc &desc) const;
  Buffer CreateBuffer(const rhi::BufferDesc &desc) const;
  Window CreateHeadlessWindow(const rhi::WindowDesc &desc = {}) const;
  Window CreateGLFWWindow(const rhi::WindowDesc &desc = {}) const;
  Queue GetQueue(rhi::QueueType type) const;
  CommandContext CreateCommandContext() const;
};

}  // namespace grassland::rhix
