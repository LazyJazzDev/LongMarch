#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_timeline.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend {

class Queue : public Object<rhi::IQueue> {
 public:
  Queue(Device *device, QueueType type);
  ~Queue() override;
  Device *Owner() const;
  virtual Timeline *GetTimeline() const = 0;
  QueueType GetType() const override;
  std::mutex &QueueMutex();
  // Called under QueueMutex, in native submission order.
  uint64_t NextSerial();

 private:
  Device *device_;
  QueueType type_;
  std::mutex mutex_;
  uint64_t serial_{};
};

}  // namespace grassland::rhi::backend
