#include "grassland/rhi/backend/vulkan/vulkan_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#if !defined(_WIN32)
#include <unistd.h>
#endif
namespace grassland::rhi::backend::vulkan {
namespace {
#if defined(_WIN32)
constexpr auto memory_handle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr auto memory_handle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}  // namespace

VulkanCUDABufferImpl::VulkanCUDABufferImpl(IVulkanDevice *device, const BufferDesc &desc, int cuda_device)
    : owner_(device),
      desc_(desc),
      mapping_(cuda_device) {
}

VulkanCUDABufferImpl::~VulkanCUDABufferImpl() {
  mapping_.Reset();
  auto device = static_cast<IVulkanDevice *>(Owner())->NativeDevice();
  if (buffer_)
    vkDestroyBuffer(device, buffer_, nullptr);
  if (memory_)
    vkFreeMemory(device, memory_, nullptr);
}

BufferDesc VulkanCUDABufferImpl::GetDesc() const {
  return desc_;
}

VkBuffer VulkanCUDABufferImpl::NativeBuffer() const {
  return buffer_;
}

void *VulkanCUDABufferImpl::GetDevicePointer() const {
  return mapping_.Pointer();
}

rhi::ICUDABuffer *VulkanCUDABufferImpl::Create(IVulkanDevice *device, const BufferDesc &desc) {
  if (!desc.size)
    return nullptr;
  const int ordinal = device->GetPhysicalDevice().cuda_device;
  if (ordinal < 0) {
    ReportError(-1, "No CUDA device matches this Vulkan adapter.");
    return nullptr;
  }
#if defined(_WIN32)
  if (!device->Procedures().GetMemoryWin32HandleKHR)
    return nullptr;
#else
  if (!device->Procedures().GetMemoryFdKHR)
    return nullptr;
#endif
  VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer.size = desc.size;
  buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (device->GetPhysicalDevice().ray_query_feature || device->GetPhysicalDevice().ray_tracing_pipeline_feature)
    buffer.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  if (device->GetPhysicalDevice().ray_tracing_pipeline_feature)
    buffer.usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  VkPhysicalDeviceExternalBufferInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
  query.usage = buffer.usage;
  query.handleType = memory_handle;
  VkExternalBufferProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
  vkGetPhysicalDeviceExternalBufferProperties(device->NativePhysicalDevice(), &query, &properties);
  if (!(properties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) {
    ReportError(-1, "CUDA buffer allocation is not exportable.");
    return nullptr;
  }
  auto object = std::unique_ptr<VulkanCUDABufferImpl>(new VulkanCUDABufferImpl(device, desc, ordinal));
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
  VkMemoryAllocateFlagsInfo address{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  if (buffer.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    address.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    dedicated.pNext = &address;
  }
  VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  export_info.handleTypes = memory_handle;
  export_info.pNext = &dedicated;
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
  auto get = device->Procedures().GetMemoryWin32HandleKHR;
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
  auto get = device->Procedures().GetMemoryFdKHR;
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

}  // namespace grassland::rhi::backend::vulkan
#endif
