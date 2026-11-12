#include "grassland/rhi/backend/vulkan/vulkan_cuda_timeline.h"

#if defined(LONGMARCH_CUDA_RUNTIME)

#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace grassland::rhi::backend::vulkan {

VulkanCUDATimelineImpl::VulkanCUDATimelineImpl(IVulkanDevice *device)
    : owner_(device),
      cuda_(device->GetPhysicalDevice().cuda_device) {
}

ObjectPtr<IVulkanTimeline> VulkanCUDATimelineImpl::Create(IVulkanDevice *device) {
  if (device->GetPhysicalDevice().cuda_device < 0)
    return {};
  auto result = ObjectPtr<VulkanCUDATimelineImpl>::Adopt(new VulkanCUDATimelineImpl(device));
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  info.pNext = &type;
#if defined(_WIN32)
  constexpr auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  constexpr auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
  VkExportSemaphoreCreateInfo external{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};

  VkPhysicalDeviceExternalSemaphoreInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
  query.pNext = &type;
  query.handleType = handle_type;
  VkExternalSemaphoreProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  vkGetPhysicalDeviceExternalSemaphoreProperties(device->NativePhysicalDevice(), &query, &properties);
  if (!(properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)) {
    ReportError(-1, "CUDA timeline semaphore is not exportable.");
    return {};
  }
  external.handleTypes = handle_type;
  type.pNext = &external;

  if (!Check(vkCreateSemaphore(device->NativeDevice(), &info, nullptr, &result->semaphore_), "Create queue timeline"))
    return {};

  cudaExternalSemaphoreHandleDesc desc{};
#if defined(_WIN32)
  auto get = device->Procedures().GetSemaphoreWin32HandleKHR;
  if (!get)
    return {};
  VkSemaphoreGetWin32HandleInfoKHR export_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  export_info.semaphore = result->semaphore_;
  export_info.handleType = handle_type;
  HANDLE handle{};
  if (!Check(get(device->NativeDevice(), &export_info, &handle), "Export CUDA timeline"))
    return {};
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  desc.handle.win32.handle = handle;
  const bool imported = result->cuda_.Import(desc);
  CloseHandle(handle);
#else
  auto get = device->Procedures().GetSemaphoreFdKHR;
  if (!get)
    return {};
  VkSemaphoreGetFdInfoKHR export_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  export_info.semaphore = result->semaphore_;
  export_info.handleType = handle_type;
  int fd = -1;
  if (!Check(get(device->NativeDevice(), &export_info, &fd), "Export CUDA timeline"))
    return {};
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  desc.handle.fd = fd;
  const bool imported = result->cuda_.Import(desc);
  if (!imported)
    close(fd);
#endif
  if (!imported)
    return {};

  return result;
}

VulkanCUDATimelineImpl::~VulkanCUDATimelineImpl() {
  cuda_.Reset();
  if (semaphore_)
    vkDestroySemaphore(owner_->NativeDevice(), semaphore_, nullptr);
}

bool VulkanCUDATimelineImpl::GetCompletedValue(uint64_t &value) const {
  return Check(vkGetSemaphoreCounterValue(owner_->NativeDevice(), semaphore_, &value), "Query timeline completion");
}

bool VulkanCUDATimelineImpl::WaitHost(uint64_t value) const {
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore_;
  info.pValues = &value;
  return Check(vkWaitSemaphores(owner_->NativeDevice(), &info, UINT64_MAX), "Wait for timeline");
}

VkSemaphore VulkanCUDATimelineImpl::NativeSemaphore() const {
  return semaphore_;
}

ObjectPtr<IVulkanTimeline> CreateVulkanCUDATimeline(IVulkanDevice *device) {
  return VulkanCUDATimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::vulkan

#endif
