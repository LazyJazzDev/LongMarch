#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendSubmissionToken)
#include <condition_variable>
#include <mutex>

#include "grassland/rhi/rhi_submission_token.h"

namespace grassland::rhi::backend {

// Completion state borrows its timeline; it owns no native resources or commands.
class IBackendSubmissionToken : public rhi::ISubmissionToken {
 public:
  static IBackendSubmissionToken *Create(Event event);
  virtual Event GetEvent() const = 0;
  virtual SubmissionStatus GetStatus() const = 0;
  virtual bool WaitHost() const = 0;
  virtual void Complete(SubmissionStatus status) = 0;
};

;

}  // namespace grassland::rhi::backend
