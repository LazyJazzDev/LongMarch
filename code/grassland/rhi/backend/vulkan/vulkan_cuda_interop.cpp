#include "grassland/rhi/backend/vulkan/vulkan_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#if defined(_WIN32)
#include <windows.h>

// Vulkan's Win32 declarations require the Windows handle types above.
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif
namespace grassland::rhi::backend::vulkan {
namespace {
#if defined(_WIN32)
constexpr auto memory_handle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr auto semaphore_handle = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr auto memory_handle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr auto semaphore_handle = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}  // namespace

VulkanCUDABuffer::VulkanCUDABuffer(VulkanDevice *device, const BufferDesc &desc)
    : VulkanBuffer(device, desc),
      mapping_(device->GetPhysicalDevice().cuda_device) {
}

VulkanCUDABuffer::~VulkanCUDABuffer() {
  mapping_.Reset();
  auto device = static_cast<VulkanDevice *>(device_)->NativeDevice();
  if (buffer_)
    vkDestroyBuffer(device, buffer_, nullptr);
  if (memory_)
    vkFreeMemory(device, memory_, nullptr);
}

VkBuffer VulkanCUDABuffer::NativeBuffer() const {
  return buffer_;
}

void *VulkanCUDABuffer::GetDevicePointer() const {
  return mapping_.Pointer();
}

VulkanCUDABuffer *VulkanCUDABuffer::Create(VulkanDevice *device, const BufferDesc &desc) {
  if (!desc.size || device->GetPhysicalDevice().cuda_device < 0)
    return nullptr;
#if defined(_WIN32)
  if (!vkGetDeviceProcAddr(device->NativeDevice(), "vkGetMemoryWin32HandleKHR"))
    return nullptr;
#else
  if (!vkGetDeviceProcAddr(device->NativeDevice(), "vkGetMemoryFdKHR"))
    return nullptr;
#endif
  VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer.size = desc.size;
  buffer.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  const bool addressable = device->SupportRayTracingPipeline() || device->SupportRayQuery();
  if (addressable) {
    buffer.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (device->SupportRayTracingPipeline())
      buffer.usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  }
  VkPhysicalDeviceExternalBufferInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
  query.usage = buffer.usage;
  query.handleType = memory_handle;
  VkExternalBufferProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
  vkGetPhysicalDeviceExternalBufferProperties(device->NativePhysicalDevice(), &query, &properties);
  if (!(properties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) {
    ReportError(-1, "CUDA buffer allocation is not exportable.");
    return nullptr;
  }
  auto object = std::unique_ptr<VulkanCUDABuffer>(new VulkanCUDABuffer(device, desc));
  VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  external.handleTypes = memory_handle;
  buffer.pNext = &external;
  if (!Check(vkCreateBuffer(device->NativeDevice(), &buffer, nullptr, &object->buffer_), "Create CUDA buffer"))
    return nullptr;
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device->NativeDevice(), object->buffer_, &requirements);
  VkPhysicalDeviceMemoryProperties types{};
  vkGetPhysicalDeviceMemoryProperties(device->NativePhysicalDevice(), &types);
  uint32_t index = UINT32_MAX;
  for (uint32_t i = 0; i < types.memoryTypeCount; ++i)
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (types.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      index = i;
      break;
    }
  if (index == UINT32_MAX)
    return nullptr;
  VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicated.buffer = object->buffer_;
  VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  export_info.handleTypes = memory_handle;
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  flags.pNext = &dedicated;
  export_info.pNext = addressable ? static_cast<void *>(&flags) : static_cast<void *>(&dedicated);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.pNext = &export_info;
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = index;
  if (!Check(vkAllocateMemory(device->NativeDevice(), &allocation, nullptr, &object->memory_),
             "Allocate CUDA memory") ||
      !Check(vkBindBufferMemory(device->NativeDevice(), object->buffer_, object->memory_, 0), "Bind CUDA memory"))
    return nullptr;
  cudaExternalMemoryHandleDesc memory{};
  memory.size = requirements.size;
  memory.flags = cudaExternalMemoryDedicated;
#if defined(_WIN32)
  VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory = object->memory_;
  info.handleType = memory_handle;
  auto get = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
      vkGetDeviceProcAddr(device->NativeDevice(), "vkGetMemoryWin32HandleKHR"));
  HANDLE handle{};
  if (!Check(get(device->NativeDevice(), &info, &handle), "Export CUDA memory"))
    return nullptr;
  memory.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  memory.handle.win32.handle = handle;
  const bool imported = object->mapping_.Import(memory, desc.size);
  CloseHandle(handle);
#else
  VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
  info.memory = object->memory_;
  info.handleType = memory_handle;
  auto get = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device->NativeDevice(), "vkGetMemoryFdKHR"));
  int fd = -1;
  if (!Check(get(device->NativeDevice(), &info, &fd), "Export CUDA memory"))
    return nullptr;
  memory.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memory.handle.fd = fd;
  const bool imported = object->mapping_.Import(memory, desc.size);
  if (!object->mapping_.HasMemory())
    close(fd);
#endif
  return imported ? object.release() : nullptr;
}

VulkanCUDAInteropTimeline::VulkanCUDAInteropTimeline(VulkanDevice *device) : CUDAInteropTimeline(device) {
}

VulkanCUDAInteropTimeline::~VulkanCUDAInteropTimeline() {
  Reset();
  if (semaphore_)
    vkDestroySemaphore(static_cast<VulkanDevice *>(Owner())->NativeDevice(), semaphore_, nullptr);
}

VkSemaphore VulkanCUDAInteropTimeline::NativeSemaphore() const {
  return semaphore_;
}

VulkanCUDAInteropTimeline *VulkanCUDAInteropTimeline::Create(VulkanDevice *device) {
  if (device->GetPhysicalDevice().cuda_device < 0)
    return nullptr;
#if defined(_WIN32)
  if (!vkGetDeviceProcAddr(device->NativeDevice(), "vkGetSemaphoreWin32HandleKHR"))
    return nullptr;
#else
  if (!vkGetDeviceProcAddr(device->NativeDevice(), "vkGetSemaphoreFdKHR"))
    return nullptr;
#endif
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkPhysicalDeviceExternalSemaphoreInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
  query.pNext = &type;
  query.handleType = semaphore_handle;
  VkExternalSemaphoreProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  vkGetPhysicalDeviceExternalSemaphoreProperties(device->NativePhysicalDevice(), &query, &properties);
  if (!(properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT))
    return nullptr;
  auto object = std::unique_ptr<VulkanCUDAInteropTimeline>(new VulkanCUDAInteropTimeline(device));
  VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
  export_info.handleTypes = semaphore_handle;
  export_info.pNext = &type;
  VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  create.pNext = &export_info;
  if (!Check(vkCreateSemaphore(device->NativeDevice(), &create, nullptr, &object->semaphore_), "Create CUDA timeline"))
    return nullptr;
  cudaExternalSemaphoreHandleDesc desc{};
#if defined(_WIN32)
  VkSemaphoreGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  info.semaphore = object->semaphore_;
  info.handleType = semaphore_handle;
  auto get = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
      vkGetDeviceProcAddr(device->NativeDevice(), "vkGetSemaphoreWin32HandleKHR"));
  HANDLE handle{};
  if (!Check(get(device->NativeDevice(), &info, &handle), "Export CUDA timeline"))
    return nullptr;
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  desc.handle.win32.handle = handle;
  const bool imported = object->Import(desc);
  CloseHandle(handle);
#else
  VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  info.semaphore = object->semaphore_;
  info.handleType = semaphore_handle;
  auto get =
      reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device->NativeDevice(), "vkGetSemaphoreFdKHR"));
  int fd = -1;
  if (!Check(get(device->NativeDevice(), &info, &fd), "Export CUDA timeline"))
    return nullptr;
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  desc.handle.fd = fd;
  const bool imported = object->Import(desc);
  if (!imported)
    close(fd);
#endif
  return imported ? object.release() : nullptr;
}

ICUDABuffer *VulkanDevice::CreateCUDABuffer(const BufferDesc &desc) {
  return VulkanCUDABuffer::Create(this, desc);
}

ICUDAInteropTimeline *VulkanDevice::CreateCUDAInteropTimeline() {
  return VulkanCUDAInteropTimeline::Create(this);
}
}  // namespace grassland::rhi::backend::vulkan
#endif
