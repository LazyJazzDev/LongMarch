#pragma once
#include "grassland/rhi/rhi_object.h"

LONGMARCH_RHI_UUID(grassland::rhi::IQueue, "688dec05-89c3-4429-bef3-8c3d5feba2d0")

namespace grassland::rhi {
class ICommandContext;

class IQueue : public IObject {
 public:
  // Creates an owned context permanently bound to this queue.
  virtual ICommandContext *CreateCommandContext() = 0;
  virtual QueueType GetType() const = 0;

 protected:
  ~IQueue() override;
};

}  // namespace grassland::rhi
