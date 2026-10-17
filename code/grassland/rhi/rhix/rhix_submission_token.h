#pragma once
#include "grassland/rhi/rhi_submission_token.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {

class SubmissionToken : public ObjectContainer {
 public:
  SubmissionToken(rhi::ISubmissionToken *object) noexcept;
  rhi::ISubmissionToken *Get() const noexcept;
  SubmissionToken() noexcept;
  rhi::SubmissionStatus GetStatus() const;
  bool WaitHost() const;
};

}  // namespace grassland::rhix
