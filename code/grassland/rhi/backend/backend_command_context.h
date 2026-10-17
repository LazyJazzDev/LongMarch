#pragma once
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_command_context.h"

namespace grassland::rhi::backend {

class CommandContext : public Object<rhi::ICommandContext> {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

  CommandContext(const void *identity);
  ~CommandContext() override;
  bool Seal();
  const void *DeviceIdentity() const;
  void MarkSubmitted();
  bool WasSubmitted() const;
  virtual SubmitResult SubmitNative() = 0;
  virtual SubmissionStatus WaitForProgress() = 0;

 protected:
  // Call before encoding each resource use. Retains each exact object once.
  bool RetainResource(rhi::IObject *resource);
  virtual bool EndRecording() = 0;
  bool recording_failed_{};
  bool sealed_{};

 private:
  const void *identity_;
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_;
  bool submitted_{};
};

}  // namespace grassland::rhi::backend
