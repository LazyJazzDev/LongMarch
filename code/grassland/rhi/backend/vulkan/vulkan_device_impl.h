#pragma once
#include "grassland/rhi/backend/backend_device_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDeviceImpl final : public Object<IVulkanDevice> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IDevice, IVulkanDevice, IBackendDevice, rhi::IDevice)
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  static IVulkanDevice *CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                           IBackendCore *core,
                                           VkPhysicalDevice physical_device,
                                           uint32_t api_version);
  ~VulkanDeviceImpl() override;
  VkDevice NativeDevice() const;
  VulkanRenderTargetPool &RenderTargets();
  VulkanDescriptorPool &DescriptorPool();
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override;
  IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  const DeviceProcedures &Procedures() const;
  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override;
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  VmaAllocator Allocator() const;
  VkPhysicalDevice NativePhysicalDevice() const;
  VkInstance Instance() const;
  uint32_t QueueFamily() const;
  bool SupportsSwapchain() const;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  VulkanTemporalBufferPool &UploadPool();
  VulkanTemporalBufferPool &DownloadPool();
  VulkanScratchPool &ScratchPool();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
  // The pool grows on demand, independently of swapchain image count.
  bool AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command);
  void RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command);

 private:
  void InterruptCollector() override;
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const override;
  void WaitForProgress(const std::vector<Event> &next) override;
  VulkanTemporalBufferPool upload_pool_;
  VulkanTemporalBufferPool download_pool_;
  VulkanScratchPool scratch_pool_;
  std::mutex resource_pool_mutex_;
  std::vector<std::tuple<VkCommandPool, VkCommandBuffer>> command_pool_;
  VulkanDeviceImpl(grassland::rhi::PhysicalDevice info, IBackendCore *core);
  std::unique_ptr<VulkanRenderTargetPool> render_targets_;
  std::unique_ptr<VulkanDescriptorPool> descriptor_pool_;
  VmaAllocator allocator_{};
  VkDevice device_{VK_NULL_HANDLE};
  DeviceProcedures procedures_;
  VkPhysicalDevice physical_{};
  uint32_t queue_family_index_{};
  bool swapchain_enabled_{};
  VkSemaphore collector_shutdown_{VK_NULL_HANDLE};
  bool shutdown_consumed_{};

 private:
  DeviceState state_Device_;

  DeviceState &StateDevice() override {
    return state_Device_;
  }

  const DeviceState &StateDevice() const override {
    return state_Device_;
  }

 public:
  std::string Name() const override {
    return IBackendDevice::CommonName();
  }

  bool SupportRayTracingPipeline() const override {
    return IBackendDevice::CommonSupportRayTracingPipeline();
  }

  bool SupportRayQuery() const override {
    return IBackendDevice::CommonSupportRayQuery();
  }

  bool SupportGeometryShader() const override {
    return IBackendDevice::CommonSupportGeometryShader();
  }

  rhi::IShader *CreateShader(const ShaderDesc &desc) override {
    return IBackendDevice::CommonCreateShader(desc);
  }

  rhi::IQueue *GetQueue(QueueType type) override {
    return IBackendDevice::CommonGetQueue(type);
  }

  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override {
    return IBackendDevice::CommonWaitEvents(events, wait_all);
  }
};
}  // namespace grassland::rhi::backend::vulkan
