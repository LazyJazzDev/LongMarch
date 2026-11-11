#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class SubmissionToken;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::SubmissionToken)
#include <condition_variable>
#include <mutex>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_submission_token.h"

namespace grassland::rhi::backend {

// Completion state borrows its timeline; it owns no native resources or commands.
class SubmissionToken final : public Object<rhi::ISubmissionToken> {
 public:
  LONGMARCH_RHI_QUERY(SubmissionToken, Object<rhi::ISubmissionToken>)

  explicit SubmissionToken(Event event);
  Event GetEvent() const override;
  SubmissionStatus GetStatus() const override;
  bool WaitHost() const override;
  void Complete(SubmissionStatus status);

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  SubmissionStatus status_{SUBMISSION_STATUS_PENDING};
  const Event event_;
};

}  // namespace grassland::rhi::backend
