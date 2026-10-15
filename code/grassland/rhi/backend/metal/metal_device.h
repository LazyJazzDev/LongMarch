#pragma once

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend::metal {

class MetalDevice final : public Device {
 public:
  MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDevice() override;

  MTL::Device *NativeDevice() const noexcept;
  bool IsInitialized() const;
  MTL::CommandQueue *NativeQueue() const;
  std::mutex &QueueMutex();
  bool WaitIdle();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

 private:
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
  std::mutex queue_mutex_;
};

}  // namespace grassland::rhi::backend::metal
