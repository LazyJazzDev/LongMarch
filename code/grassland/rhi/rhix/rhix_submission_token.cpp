#include "grassland/rhi/rhix/rhix_submission_token.h"

namespace grassland::rhix {

SubmissionToken::SubmissionToken() noexcept = default;

SubmissionToken::SubmissionToken(rhi::ISubmissionToken *object) noexcept : ObjectContainer(object) {
}

rhi::ISubmissionToken *SubmissionToken::Get() const noexcept {
  return static_cast<rhi::ISubmissionToken *>(GetObject());
}

rhi::SubmissionStatus SubmissionToken::GetStatus() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::SUBMISSION_STATUS_CANCELED;
#endif
  return Get()->GetStatus();
}

bool SubmissionToken::WaitHost() const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->WaitHost();
}

}  // namespace grassland::rhix
