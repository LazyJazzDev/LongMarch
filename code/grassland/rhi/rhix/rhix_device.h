#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhix/rhix_command_context.h"
#include "grassland/rhi/rhix/rhix_object_container.h"
#include "grassland/rhi/rhix/rhix_queue.h"
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
  Window CreateWindow(const rhi::WindowDesc &desc = {}) const;
  Queue GetQueue(rhi::QueueType type) const;
  CommandContext CreateCommandContext(const Queue &queue) const;
};

}  // namespace grassland::rhix
