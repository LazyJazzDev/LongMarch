#pragma once
#include <tuple>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_device_procedures.h"
#include "grassland/rhi/backend/vulkan/vulkan_render_target_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_scratch_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanDevice)

namespace grassland::rhi::backend::vulkan {
class IVulkanDevice : public IBackendDevice {
 public:
  static IVulkanDevice *CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                           IBackendCore *core,
                                           VkPhysicalDevice physical_device,
                                           uint32_t api_version);
  virtual IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) = 0;
  virtual IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) = 0;
  virtual ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) = 0;
  virtual VkDevice NativeDevice() const = 0;
  virtual VulkanRenderTargetPool &RenderTargets() = 0;
  virtual VulkanDescriptorPool &DescriptorPool() = 0;
  virtual ISampler *CreateSampler(const SamplerDesc &desc = {}) = 0;
  virtual IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) = 0;
  virtual const DeviceProcedures &Procedures() const = 0;
  virtual rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) = 0;
  virtual rhi::IProgram *CreateProgram(const ProgramDesc &desc) = 0;
  virtual VmaAllocator Allocator() const = 0;
  virtual VkPhysicalDevice NativePhysicalDevice() const = 0;
  virtual VkInstance Instance() const = 0;
  virtual uint32_t QueueFamily() const = 0;
  virtual bool SupportsSwapchain() const = 0;
  virtual rhi::IImage *CreateImage(const ImageDesc &desc) = 0;
  virtual rhi::IBuffer *CreateBuffer(const BufferDesc &desc) = 0;
  virtual VulkanTemporalBufferPool &UploadPool() = 0;
  virtual VulkanTemporalBufferPool &DownloadPool() = 0;
  virtual VulkanScratchPool &ScratchPool() = 0;
  virtual rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) = 0;
  virtual  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
      // The pool grows on demand, independently of swapchain image count.
      bool
      AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command) = 0;
  virtual void RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command) = 0;
};

;
}  // namespace grassland::rhi::backend::vulkan
