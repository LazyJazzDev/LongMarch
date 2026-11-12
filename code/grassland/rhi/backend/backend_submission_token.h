#pragma once
#include <condition_variable>
#include <mutex>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_submission_token.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendSubmissionToken,
                        grassland::rhi::IObject,
                        "92c56d21-d1e3-4070-a232-2cc80606c437")

namespace grassland::rhi::backend {

// Completion state borrows its timeline; it owns no native resources or commands.
class IBackendSubmissionToken : public IObject {
 public:
  static IBackendSubmissionToken *Create(Event event);
  virtual void Complete(SubmissionStatus status) = 0;
};

}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend {
class SubmissionTokenImpl final : public Object<rhi::ISubmissionToken, IBackendSubmissionToken> {
 public:
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
