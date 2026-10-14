#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {

class IQueue : public IObject {
 public:
  IQueue(IDevice *device, QueueType type);
  ~IQueue() override;
  IDevice *Owner() const;
  QueueType Type() const;
  virtual std::unique_ptr<ICommandContext> CreateCommandContext() const = 0;

 private:
  IDevice *device_;
  QueueType type_;
};

}  // namespace grassland::rhi::backend
