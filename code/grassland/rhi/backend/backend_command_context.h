#pragma once
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendCommandContext, "4a13943e-6f88-4739-91c4-0eb6b6a04a0c")

namespace grassland::rhi::backend {
class IBackendCommandContext : public IObject {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

 protected:
  void DestroyCommandContext();

 public:
  virtual void BindDownloads(IBackendSubmissionToken *submission) = 0;
  virtual void ReleaseDownloads() = 0;
  virtual bool Seal() = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual IBackendQueue *OwnerQueue() const = 0;
  virtual void MarkSubmitted() = 0;
  virtual bool WasSubmitted() const = 0;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;

 protected:
  virtual bool EndRecording() = 0;
};

}  // namespace grassland::rhi::backend
