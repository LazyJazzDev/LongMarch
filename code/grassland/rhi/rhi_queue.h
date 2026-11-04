#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class ICommandContext;

class IQueue : public IObject {
 public:
  virtual QueueType GetType() const = 0;
  // Use this factory for contexts intended for this queue (required by Metal).
  virtual ICommandContext *CreateCommandContext() = 0;
  virtual bool SharesPhysicalQueue(const IQueue *other) const = 0;

 protected:
  ~IQueue() override;
};

}  // namespace grassland::rhi
