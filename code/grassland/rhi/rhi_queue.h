#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class ICommandContext;

class ISubmissionToken : public IObject {
 public:
  virtual SubmissionStatus GetStatus() const = 0;
  virtual bool WaitHost() const = 0;

 protected:
  ~ISubmissionToken() override;
};

struct SubmitInfo {
  // Borrowed for the duration of Submit.
  std::vector<ISubmissionToken *> waits;
};

class IQueue : public IObject {
 public:
  // Borrows the context and ends recording. Accepted work retains its resources
  // until GPU completion. Submitted contexts cannot record or submit again.
  // Returns an owned completion token, or nullptr on failure.
  virtual ISubmissionToken *Submit(ICommandContext *commands, const SubmitInfo &info = {}) = 0;
  virtual QueueType GetType() const = 0;
  virtual bool SharesPhysicalQueue(const IQueue *other) const = 0;

 protected:
  ~IQueue() override;
};

}  // namespace grassland::rhi
