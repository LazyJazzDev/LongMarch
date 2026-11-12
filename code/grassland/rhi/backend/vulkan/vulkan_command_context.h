#pragma once
#include <array>
#include <unordered_map>
#include <vector>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanCommandContext : public IObject {
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
  virtual IVulkanQueue *OwnerQueue() const = 0;
  virtual void MarkSubmitted() = 0;
  virtual bool WasSubmitted() const = 0;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;
};

ICommandContext *CreateVulkanCommandContext(IVulkanQueue *queue);

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanCommandContext,
                        grassland::rhi::IObject,
                        "c66f4335-1f50-47ca-9a8d-754583fbdfea")
