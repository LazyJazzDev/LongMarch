#include "grassland/rhi/rhix/rhix_submission_token.h"

namespace grassland::rhix {
rhi::Event SubmissionToken::GetEvent() const {
  return *this ? Get()->GetEvent() : rhi::Event{};
}

SubmissionToken::SubmissionToken() noexcept = default;

SubmissionToken::SubmissionToken(rhi::ISubmissionToken *object) noexcept : ObjectContainer(object) {
}

rhi::ISubmissionToken *SubmissionToken::Get() const noexcept {
  return rhi::QueryInterfacePtr<rhi::ISubmissionToken>(GetObject()).Get();
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
