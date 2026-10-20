#pragma once

#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/metal/metal_upload_pool.h"

namespace grassland::rhi::backend::metal {

class MetalDevice final : public Device {
 public:
  static MetalDevice *Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDevice() override;

  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  MetalUploadPool &UploadPool();
  MTL::Device *NativeDevice() const noexcept;
  MTL::CommandQueue *NativeQueue() const;
  std::mutex &QueueMutex();
  bool WaitIdle();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

 private:
  MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);

  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
  std::mutex queue_mutex_;
  MetalUploadPool uploads_{this};
};

}  // namespace grassland::rhi::backend::metal
