#pragma once
#include <atomic>
#include <mutex>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend {
class Device;
class CommandContext;

class Queue : public Object<rhi::IQueue> {
 public:
  Queue(Device *device, QueueType type);
  ~Queue() override;
  Device *Owner() const;
  QueueType GetType() const override;
  ICommandContext *CreateCommandContext() override = 0;
  bool SharesPhysicalQueue(const rhi::IQueue *other) const override;
  ISubmissionToken *Submit(CommandContext *commands, const SubmitInfo &info);
  virtual const void *NativeIdentity() const;
  virtual std::mutex &Mutex();
  // Called while holding Mutex(), including internal native submissions.
  uint64_t NextSerial();
  virtual bool HasNativeTimeline() const;
  virtual bool QueryTimeline(uint64_t &value) const;
  void RecordCompletion(uint64_t value);

 private:
  Device *device_;
  QueueType type_;
  std::mutex mutex_;
  uint64_t serial_{};
  std::atomic<uint64_t> completed_{};
};
}  // namespace grassland::rhi::backend
