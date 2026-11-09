#pragma once
#include <condition_variable>
#include <mutex>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_submission_token.h"

namespace grassland::rhi::backend {

class Device;

// Completion state borrows its device; it owns no native resources or commands.
class SubmissionToken : public Object<rhi::ISubmissionToken> {
 public:
  explicit SubmissionToken(Device *device);
  SubmissionStatus CompletionStatus() const;
  bool WaitForCompletion() const;
  void Complete(SubmissionStatus status);
  Device *Owner() const;

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  SubmissionStatus status_{SUBMISSION_STATUS_PENDING};
  Device *device_;
};

}  // namespace grassland::rhi::backend
