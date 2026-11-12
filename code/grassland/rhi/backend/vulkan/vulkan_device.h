#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>

#include "grassland/rhi/backend/vulkan/vulkan_binding_group.h"
#include "grassland/rhi/backend/vulkan/vulkan_device_procedures.h"
#include "grassland/rhi/backend/vulkan/vulkan_render_target_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_scratch_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_device.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanDevice,
                        grassland::rhi::IObject,
                        "96a1c412-20e1-40bc-b542-d515f9a41965")

namespace grassland::rhi::backend::vulkan {
class IVulkanDevice : public IObject {
 public:
  virtual const PhysicalDevice &GetPhysicalDevice() const noexcept = 0;
  // Borrowed queue owned by this device.
  virtual IVulkanQueue *GetNativeQueue(QueueType type) const = 0;
  static rhi::IDevice *CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                          IVulkanCore *core,
                                          VkPhysicalDevice physical_device,
                                          uint32_t api_version);
  virtual VkDevice NativeDevice() const = 0;
  virtual VulkanRenderTargetPool &RenderTargets() = 0;
  virtual VulkanDescriptorPool &DescriptorPool() = 0;
  virtual const DeviceProcedures &Procedures() const = 0;
  virtual VmaAllocator Allocator() const = 0;
  virtual VkPhysicalDevice NativePhysicalDevice() const = 0;
  virtual VkInstance Instance() const = 0;
  virtual uint32_t QueueFamily() const = 0;
  virtual bool SupportsSwapchain() const = 0;
  virtual VulkanTemporalBufferPool &UploadPool() = 0;
  virtual VulkanTemporalBufferPool &DownloadPool() = 0;
  virtual VulkanScratchPool &ScratchPool() = 0;
  virtual  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
      // The pool grows on demand, independently of swapchain image count.
      bool
      AcquireCommandResources(VkCommandPool &pool, VkCommandBuffer &command) = 0;
  virtual void RecycleCommandResources(VkCommandPool pool, VkCommandBuffer command) = 0;
};

}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend::vulkan {
class VulkanDeviceImpl final : public Object<rhi::IDevice, IVulkanDevice> {
 public:
  IVulkanQueue *GetNativeQueue(QueueType type) const override;
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  static rhi::IDevice *CreateVulkanDevice(grassland::rhi::PhysicalDevice info,
                                          IVulkanCore *core,
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
  void InterruptCollector();
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const;
  void WaitForProgress(const std::vector<Event> &next);
  VulkanTemporalBufferPool upload_pool_;
  VulkanTemporalBufferPool download_pool_;
  VulkanScratchPool scratch_pool_;
  std::mutex resource_pool_mutex_;
  std::vector<std::tuple<VkCommandPool, VkCommandBuffer>> command_pool_;
  VulkanDeviceImpl(grassland::rhi::PhysicalDevice info, IVulkanCore *core);
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

  void InitializeDevice(PhysicalDevice info, IVulkanCore *core, BackendAPI api = BACKEND_API_DEFAULT);

 public:
  rhi::IShader *CreateShader(const ShaderDesc &desc) override;

 private:
  void DestroyDevice();

 public:
  std::string Name() const override;

  bool SupportRayTracingPipeline() const override;

  bool SupportRayQuery() const override;

  bool SupportGeometryShader() const override;

  rhi::IQueue *GetQueue(QueueType type) override;

  const PhysicalDevice &GetPhysicalDevice() const noexcept override;

 private:
  IVulkanCore *OwnerCore() const;

  // Called once after native device resources and all queues are initialized.
  void StartCollector();

 public:
  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override;

 private:
  void Collect();

  void ShutdownQueues();

  std::array<ObjectPtr<IVulkanQueue>, 2> queues_{};
  std::atomic<bool> stopping_{};
  BackendAPI api_{};
  IVulkanCore *core_{};
  PhysicalDevice info_{};
  std::thread collector_{};
};
}  // namespace grassland::rhi::backend::vulkan
