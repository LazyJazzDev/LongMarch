#pragma once
#include "grassland/rhi/rhi_cuda_interop.h"
#include "grassland/rhi/rhi_submission_token.h"

namespace grassland::rhi {
class ICommandContext;

struct SubmitInfo {
  // Borrowed for the duration of Submit.
  std::vector<ISubmissionToken *> waits;
#if defined(LONGMARCH_CUDA_RUNTIME)
  // Borrowed during Submit; accepted work retains timelines until completion.
  std::vector<CUDAInteropPoint> cuda_waits;
  std::vector<CUDAInteropPoint> cuda_signals;
#endif
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
