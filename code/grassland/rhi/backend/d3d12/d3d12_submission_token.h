#pragma once
#include "grassland/rhi/backend/backend_submission_token.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12SubmissionToken final : public SubmissionToken {
 public:
  SubmissionStatus GetStatus() const override {
    return CompletionStatus();
  }

  bool WaitHost() const override {
    return WaitForCompletion();
  }

  using SubmissionToken::SubmissionToken;
};
}  // namespace grassland::rhi::backend::d3d12
