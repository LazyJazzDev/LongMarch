#pragma once
#include "grassland/rhi/rhi_queue.h"
#include "grassland/rhi/rhix/rhix_submission_token.h"

namespace grassland::rhix {
class CommandContext;

class Queue : public ObjectContainer {
 public:
  Queue(rhi::IQueue *object) noexcept;
  rhi::IQueue *Get() const noexcept;
  Queue() noexcept;
  rhi::QueueType GetType() const;
  CommandContext CreateCommandContext() const;
  bool SharesPhysicalQueue(const Queue &other) const;
};

}  // namespace grassland::rhix
