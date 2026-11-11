#pragma once
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
class SubmissionTokenImpl final : public Object<IBackendSubmissionToken> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ISubmissionToken, IBackendSubmissionToken, rhi::ISubmissionToken)
  explicit SubmissionTokenImpl(Event event);
  Event GetEvent() const override;
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  void Complete(SubmissionStatus status);

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  SubmissionStatus status_{SUBMISSION_STATUS_PENDING};
  const Event event_;

 public:
  ~SubmissionTokenImpl() override = default;
};
}  // namespace grassland::rhi::backend
