#pragma once
#include "grassland/rhi/rhi_queue.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {

class CommandContext;

class Queue : public ObjectContainer {
 public:
  Queue(rhi::IQueue *object) noexcept;
  rhi::IQueue *Get() const noexcept;
  Queue() noexcept;
  CommandContext CreateCommandContext() const;
  rhi::QueueType GetType() const;
};

}  // namespace grassland::rhix
