#pragma once
#include "grassland/rhi/backend/backend_submission_token.h"

namespace grassland::rhi::backend::metal {
class MetalSubmissionToken final : public SubmissionToken {
 public:
  SubmissionStatus GetStatus() const override {
    return CompletionStatus();
  }

  bool WaitHost() const override {
    return WaitForCompletion();
  }

  using SubmissionToken::SubmissionToken;
};
}  // namespace grassland::rhi::backend::metal
