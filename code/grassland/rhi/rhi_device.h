#pragma once
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_queue.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhi_window.h"

namespace grassland::rhi {

class Device {
 public:
  std::string Name() const;
  bool SupportRayTracingPipeline() const;
  bool SupportRayQuery() const;
  bool SupportGeometryShader() const;
  Window CreateWindow(const WindowDesc &desc = {}) const;
  Queue GetQueue(QueueType type) const;
  CommandContext CreateCommandContext(const Queue &queue) const;
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::IDevice;
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
