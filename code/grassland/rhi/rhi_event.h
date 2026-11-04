#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
// An immutable point on a device timeline. Signal may still be pending when the
// event is returned. Events borrow their device and retain no submitted payload.
class IEvent : public IObject {
 public:
  // Only synchronization progress/device failure, not the submission result.
  virtual SubmissionStatus GetStatus() const = 0;
  virtual bool WaitHost() const = 0;

 protected:
  ~IEvent() override;
};
}  // namespace grassland::rhi
