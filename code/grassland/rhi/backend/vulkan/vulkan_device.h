#pragma once
#include <tuple>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_render_target_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_scratch_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_upload_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice final : public Device {
 public:
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  static VulkanDevice *CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                          Core *core,
                                          VkPhysicalDevice physical_device,
                                          uint32_t api_version);
  ~VulkanDevice() override;
  VkDevice NativeDevice() const;
  VulkanRenderTargetPool &RenderTargets();
  VulkanDescriptorPool &DescriptorPool();
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override;
  IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  PFN_vkCmdBeginRendering BeginRenderingCommand() const;
  PFN_vkCmdEndRendering EndRenderingCommand() const;
  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override;
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  VmaAllocator Allocator() const;
  VkPhysicalDevice NativePhysicalDevice() const;
  VkInstance Instance() const;
  VkQueue NativeQueue() const;
  uint32_t QueueFamily() const;
  bool SupportsSwapchain() const;
  std::mutex &QueueMutex();
  VkSemaphore Timeline() const;
  // Allocate and submit serials while holding QueueMutex, preserving signal order.
  uint64_t NextSerial();
  VkResult Wait(uint64_t value, uint64_t timeout = UINT64_MAX) const;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  VulkanUploadPool &UploadPool();
  VulkanScratchPool &ScratchPool();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
  // The pool grows on demand, independently of swapchain image count.
  bool AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command);
  void RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command);

 private:
  VulkanUploadPool upload_pool_;
  VulkanScratchPool scratch_pool_;
  std::mutex resource_pool_mutex_;
  std::vector<std::tuple<VkCommandPool, VkCommandBuffer>> command_pool_;
  VulkanDevice(grassland::rhi::PhysicalDevice info, Core *core);
  std::unique_ptr<VulkanRenderTargetPool> render_targets_;
  std::unique_ptr<VulkanDescriptorPool> descriptor_pool_;
  VmaAllocator allocator_{};
  VkDevice device_{VK_NULL_HANDLE};
  PFN_vkCmdBeginRendering begin_rendering_{};
  PFN_vkCmdEndRendering end_rendering_{};
  VkPhysicalDevice physical_{};
  VkQueue queue_{};
  VkSemaphore timeline_{};
  uint64_t next_serial_{};
  uint32_t graphics_queue_family_index_{};
  bool swapchain_enabled_{};
  std::mutex mutex_;
};
}  // namespace grassland::rhi::backend::vulkan
