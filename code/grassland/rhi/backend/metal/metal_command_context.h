#pragma once
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend::metal {
class IMetalCommandContext : public IObject {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

 public:
  virtual void BindDownloads(rhi::ISubmissionToken *submission) = 0;
  virtual void ReleaseDownloads() = 0;
  virtual bool Seal() = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual IMetalQueue *OwnerQueue() const = 0;
  virtual void MarkSubmitted() = 0;
  virtual bool WasSubmitted() const = 0;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;
};

ICommandContext *CreateMetalCommandContext(IMetalQueue *queue);

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalCommandContext,
                        grassland::rhi::IObject,
                        "780df097-069f-45ea-ba33-7ec47b6c22b4")
