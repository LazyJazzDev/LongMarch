#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {

class IQueue : public IObject {
 public:
  IQueue(const ObjectPtr<IDevice> &device, QueueType type);
  ~IQueue() override;
  const ObjectPtr<IDevice> &Owner() const;
  QueueType Type() const;
  virtual std::unique_ptr<ICommandContext> CreateCommandContext() const = 0;

 private:
  ObjectPtr<IDevice> device_;
  QueueType type_;
};

}  // namespace grassland::rhi::backend
