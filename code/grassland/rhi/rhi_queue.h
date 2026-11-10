#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class ICommandContext;

class IQueue : public IObject {
 public:
  // Creates an owned context permanently bound to this queue.
  virtual ICommandContext *CreateCommandContext() = 0;
  virtual QueueType GetType() const = 0;
  virtual bool SharesPhysicalQueue(const IQueue *other) const = 0;

 protected:
  ~IQueue() override;
};

}  // namespace grassland::rhi
