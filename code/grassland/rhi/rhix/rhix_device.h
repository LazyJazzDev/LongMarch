#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhix/rhix_buffer.h"
#include "grassland/rhi/rhix/rhix_command_context.h"
#include "grassland/rhi/rhix/rhix_object_container.h"
#include "grassland/rhi/rhix/rhix_program.h"
#include "grassland/rhi/rhix/rhix_queue.h"
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
  Shader CreateShader(const rhi::ShaderDesc &desc) const;
  Program CreateProgram(const rhi::ProgramDesc &desc) const;
  Buffer CreateBuffer(const rhi::BufferDesc &desc) const;
  Window CreateGLFWWindow(const rhi::WindowDesc &desc = {}) const;
  Queue GetQueue(rhi::QueueType type) const;
  CommandContext CreateCommandContext() const;
};

}  // namespace grassland::rhix
