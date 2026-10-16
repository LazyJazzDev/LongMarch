#pragma once
#include <condition_variable>
#include <mutex>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_submission_token.h"

namespace grassland::rhi::backend {

// Lightweight completion state: deliberately holds no native resources or packet.
class SubmissionToken final : public Object<rhi::ISubmissionToken> {
 public:
  explicit SubmissionToken(uint64_t queue_identity);
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  void Complete(SubmissionStatus status);
  uint64_t QueueIdentity() const;

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  SubmissionStatus status_{SUBMISSION_STATUS_PENDING};
  uint64_t queue_identity_;
};

}  // namespace grassland::rhi::backend
