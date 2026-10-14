#pragma once
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_object_container.h"
#include "grassland/rhi/rhi_queue.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhi_window.h"

namespace grassland::rhi {

class Device : public ObjectContainer {
 public:
  Device() noexcept;
  std::string Name() const;
  bool SupportRayTracingPipeline() const;
  bool SupportRayQuery() const;
  bool SupportGeometryShader() const;
  Window CreateWindow(const WindowDesc &desc = {}) const;
  Queue GetQueue(QueueType type) const;
  CommandContext CreateCommandContext(const Queue &queue) const;

 private:
  using Interface = backend::IDevice;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
