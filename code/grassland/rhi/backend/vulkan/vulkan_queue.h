#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_submission_token.h"

namespace grassland::rhi::backend::vulkan {

class VulkanQueue final : public Queue {
 public:
  rhi::ISubmissionToken *Submit(rhi::ICommandContext *commands, const rhi::SubmitInfo &info = {}) override {
    return SubmitCommands(commands, info,
                          ObjectPtr<backend::SubmissionToken>::Adopt(new VulkanSubmissionToken(Owner())));
  }

  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  QueueType GetType() const override {
    return QueueRole();
  }

  bool SharesPhysicalQueue(const rhi::IQueue *other) const override {
    return UsesSameDeviceQueue(other);
  }

  VulkanQueue(VulkanDevice *device, QueueType type);
};

}  // namespace grassland::rhi::backend::vulkan
