#include "grassland/rhi/backend/vulkan/vulkan_acceleration_structure.h"

#include <vk_mem_alloc.h>

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
void VulkanBottomLevelAccelerationStructureImpl::InitializeVulkanAccelerationStructure(IVulkanDevice *device) {
  device_ = device;
}

void VulkanTopLevelAccelerationStructureImpl::InitializeVulkanAccelerationStructure(IVulkanDevice *device) {
  device_ = device;
}

IVulkanDevice *VulkanBottomLevelAccelerationStructureImpl::Owner() const {
  return device_;
}

IVulkanDevice *VulkanTopLevelAccelerationStructureImpl::Owner() const {
  return device_;
}

uint64_t VulkanBottomLevelAccelerationStructureImpl::ScratchSize(AccelerationStructureBuildMode mode) const {
  return mode == ACCELERATION_STRUCTURE_UPDATE ? update_scratch_ : build_scratch_;
}

uint64_t VulkanTopLevelAccelerationStructureImpl::ScratchSize(AccelerationStructureBuildMode mode) const {
  return mode == ACCELERATION_STRUCTURE_UPDATE ? update_scratch_ : build_scratch_;
}

bool VulkanBottomLevelAccelerationStructureImpl::AllowsUpdate() const {
  return allow_update_;
}

bool VulkanTopLevelAccelerationStructureImpl::AllowsUpdate() const {
  return allow_update_;
}

bool VulkanBottomLevelAccelerationStructureImpl::AllowsCompaction() const {
  return allow_compaction_;
}

bool VulkanTopLevelAccelerationStructureImpl::AllowsCompaction() const {
  return allow_compaction_;
}

bool VulkanBottomLevelAccelerationStructureImpl::IsCompactStorage() const {
  return compact_;
}

bool VulkanTopLevelAccelerationStructureImpl::IsCompactStorage() const {
  return compact_;
}

VulkanBottomLevelAccelerationStructureImpl::VulkanBottomLevelAccelerationStructureImpl(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc)
    : desc_(desc) {
  InitializeVulkanAccelerationStructure(device);
}

const BottomLevelAccelerationStructureDesc &VulkanBottomLevelAccelerationStructureImpl::GetDesc() const {
  return desc_;
}

AccelerationStructureType VulkanBottomLevelAccelerationStructureImpl::GetType() const {
  return ACCELERATION_STRUCTURE_BOTTOM_LEVEL;
}

uint64_t VulkanBottomLevelAccelerationStructureImpl::GetCompactedSize() const {
  return ReadCompactedSize();
}

VulkanTopLevelAccelerationStructureImpl::VulkanTopLevelAccelerationStructureImpl(
    IVulkanDevice *device,
    const TopLevelAccelerationStructureDesc &desc)
    : desc_(desc) {
  InitializeVulkanAccelerationStructure(device);
}

const TopLevelAccelerationStructureDesc &VulkanTopLevelAccelerationStructureImpl::GetDesc() const {
  return desc_;
}

AccelerationStructureType VulkanTopLevelAccelerationStructureImpl::GetType() const {
  return ACCELERATION_STRUCTURE_TOP_LEVEL;
}

uint64_t VulkanTopLevelAccelerationStructureImpl::GetCompactedSize() const {
  return ReadCompactedSize();
}

const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &VulkanTopLevelAccelerationStructureImpl::BottomLevels()
    const {
  return bottom_levels_;
}

void VulkanTopLevelAccelerationStructureImpl::CommitDependencies(
    std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) {
  bottom_levels_ = std::move(references);
}

void VulkanBottomLevelAccelerationStructureImpl::DestroyVulkanAccelerationStructure() {
  auto device = device_->NativeDevice();
  if (query_)
    vkDestroyQueryPool(device, query_, nullptr);
  if (structure_)
    device_->Procedures().DestroyAccelerationStructureKHR(device, structure_, nullptr);
  if (buffer_)
    vmaDestroyBuffer(device_->Allocator(), buffer_, allocation_);
}

void VulkanTopLevelAccelerationStructureImpl::DestroyVulkanAccelerationStructure() {
  auto device = device_->NativeDevice();
  if (query_)
    vkDestroyQueryPool(device, query_, nullptr);
  if (structure_)
    device_->Procedures().DestroyAccelerationStructureKHR(device, structure_, nullptr);
  if (buffer_)
    vmaDestroyBuffer(device_->Allocator(), buffer_, allocation_);
}

VkAccelerationStructureKHR VulkanBottomLevelAccelerationStructureImpl::NativeStructure() const {
  return structure_;
}

VkAccelerationStructureKHR VulkanTopLevelAccelerationStructureImpl::NativeStructure() const {
  return structure_;
}

VkQueryPool VulkanBottomLevelAccelerationStructureImpl::CompactionQuery() const {
  return query_;
}

VkQueryPool VulkanTopLevelAccelerationStructureImpl::CompactionQuery() const {
  return query_;
}

VkBuildAccelerationStructureFlagsKHR VulkanBottomLevelAccelerationStructureImpl::BuildFlags() const {
  return flags_;
}

VkBuildAccelerationStructureFlagsKHR VulkanTopLevelAccelerationStructureImpl::BuildFlags() const {
  return flags_;
}

uint64_t VulkanBottomLevelAccelerationStructureImpl::Address() const {
  return address_;
}

uint64_t VulkanTopLevelAccelerationStructureImpl::Address() const {
  return address_;
}

uint64_t VulkanBottomLevelAccelerationStructureImpl::ReadCompactedSize() const {
  uint64_t result{};
  if (!query_)
    return 0;
  const auto status = vkGetQueryPoolResults(device_->NativeDevice(), query_, 0, 1, sizeof(result), &result,
                                            sizeof(result), VK_QUERY_RESULT_64_BIT);
  return status == VK_SUCCESS ? result : 0;
}

uint64_t VulkanTopLevelAccelerationStructureImpl::ReadCompactedSize() const {
  uint64_t result{};
  if (!query_)
    return 0;
  const auto status = vkGetQueryPoolResults(device_->NativeDevice(), query_, 0, 1, sizeof(result), &result,
                                            sizeof(result), VK_QUERY_RESULT_64_BIT);
  return status == VK_SUCCESS ? result : 0;
}

bool VulkanBottomLevelAccelerationStructureImpl::Allocate(const VkAccelerationStructureBuildGeometryInfoKHR &info,
                                                          const std::vector<uint32_t> &counts,
                                                          uint64_t compacted_size) {
  auto device = device_->NativeDevice();
  VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 physical{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &properties};
  vkGetPhysicalDeviceProperties2(device_->NativePhysicalDevice(), &physical);
  uint64_t primitives = 0;
  for (auto count : counts)
    primitives += count;
  if (info.geometryCount > properties.maxGeometryCount)
    return false;
  if (info.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
    if (primitives > properties.maxInstanceCount)
      return false;
  } else if (primitives > properties.maxPrimitiveCount)
    return false;
  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  device_->Procedures().GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                              &info, counts.data(), &sizes);
  if (!sizes.accelerationStructureSize)
    return false;
  flags_ = info.flags;
  allow_update_ = (flags_ & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0;
  allow_compaction_ = (flags_ & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) != 0;
  compact_ = compacted_size != 0;
  if (compact_ && !allow_compaction_)
    return false;
  build_scratch_ = sizes.buildScratchSize;
  update_scratch_ = sizes.updateScratchSize;
  VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer.size = compacted_size ? compacted_size : sizes.accelerationStructureSize;
  buffer.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  device_->ConfigureSharing(buffer);
  VmaAllocationCreateInfo memory{};
  memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (!Check(vmaCreateBuffer(device_->Allocator(), &buffer, &memory, &buffer_, &allocation_, nullptr),
             "Allocate acceleration structure"))
    return false;
  VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  create.type = info.type;
  create.buffer = buffer_;
  create.size = buffer.size;
  if (!Check(device_->Procedures().CreateAccelerationStructureKHR(device, &create, nullptr, &structure_),
             "Create acceleration structure"))
    return false;
  VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  address.accelerationStructure = structure_;
  address_ = device_->Procedures().GetAccelerationStructureDeviceAddressKHR(device, &address);
  if (allow_compaction_) {
    VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    query.queryCount = 1;
    if (!Check(vkCreateQueryPool(device, &query, nullptr, &query_), "Create acceleration structure size query"))
      return false;
  }
  return true;
}

bool VulkanTopLevelAccelerationStructureImpl::Allocate(const VkAccelerationStructureBuildGeometryInfoKHR &info,
                                                       const std::vector<uint32_t> &counts,
                                                       uint64_t compacted_size) {
  auto device = device_->NativeDevice();
  VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 physical{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &properties};
  vkGetPhysicalDeviceProperties2(device_->NativePhysicalDevice(), &physical);
  uint64_t primitives = 0;
  for (auto count : counts)
    primitives += count;
  if (info.geometryCount > properties.maxGeometryCount)
    return false;
  if (info.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
    if (primitives > properties.maxInstanceCount)
      return false;
  } else if (primitives > properties.maxPrimitiveCount)
    return false;
  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  device_->Procedures().GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                              &info, counts.data(), &sizes);
  if (!sizes.accelerationStructureSize)
    return false;
  flags_ = info.flags;
  allow_update_ = (flags_ & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0;
  allow_compaction_ = (flags_ & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) != 0;
  compact_ = compacted_size != 0;
  if (compact_ && !allow_compaction_)
    return false;
  build_scratch_ = sizes.buildScratchSize;
  update_scratch_ = sizes.updateScratchSize;
  VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer.size = compacted_size ? compacted_size : sizes.accelerationStructureSize;
  buffer.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  device_->ConfigureSharing(buffer);
  VmaAllocationCreateInfo memory{};
  memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (!Check(vmaCreateBuffer(device_->Allocator(), &buffer, &memory, &buffer_, &allocation_, nullptr),
             "Allocate acceleration structure"))
    return false;
  VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  create.type = info.type;
  create.buffer = buffer_;
  create.size = buffer.size;
  if (!Check(device_->Procedures().CreateAccelerationStructureKHR(device, &create, nullptr, &structure_),
             "Create acceleration structure"))
    return false;
  VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  address.accelerationStructure = structure_;
  address_ = device_->Procedures().GetAccelerationStructureDeviceAddressKHR(device, &address);
  if (allow_compaction_) {
    VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    query.queryCount = 1;
    if (!Check(vkCreateQueryPool(device, &query, nullptr, &query_), "Create acceleration structure size query"))
      return false;
  }
  return true;
}

std::vector<VkAccelerationStructureGeometryKHR> NativeGeometries(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers) {
  auto address = [device](const BufferView &view) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = rhi::QueryInterfacePtr<IVulkanBuffer>(view.buffer).Get()->NativeBuffer();
    return vkGetBufferDeviceAddress(device->NativeDevice(), &info) + view.offset;
  };
  std::vector<VkAccelerationStructureGeometryKHR> result;
  for (size_t i = 0; i < desc.geometries.size(); ++i) {
    const auto &g = desc.geometries[i];
    VkAccelerationStructureGeometryKHR native{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    native.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    native.flags = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    auto &tri = native.geometry.triangles;
    tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.maxVertex = g.vertex_count - 1;
    tri.vertexStride = g.vertex_stride;
    tri.indexType = !g.index_count                          ? VK_INDEX_TYPE_NONE_KHR
                    : g.index_format == INDEX_FORMAT_UINT16 ? VK_INDEX_TYPE_UINT16
                                                            : VK_INDEX_TYPE_UINT32;
    if (!buffers.empty()) {
      tri.vertexData.deviceAddress = address(buffers[i].data);
      if (g.index_count)
        tri.indexData.deviceAddress = address(buffers[i].indices);
    }
    result.push_back(native);
  }
  for (size_t i = 0; i < desc.aabbs.size(); ++i) {
    const auto &g = desc.aabbs[i];
    VkAccelerationStructureGeometryKHR native{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    native.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    native.flags = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    native.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    native.geometry.aabbs.stride = g.stride;
    if (!buffers.empty())
      native.geometry.aabbs.data.deviceAddress = address(buffers[i].data);
    result.push_back(native);
  }
  return result;
}

IBottomLevelAccelerationStructure *VulkanBottomLevelAccelerationStructureImpl::Create(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc) {
  if (!device->GetPhysicalDevice().ray_tracing_pipeline_feature && !device->GetPhysicalDevice().ray_query_feature)
    return nullptr;
  if (!ValidateBottomLevelDesc(desc))
    return nullptr;
  auto geometries = NativeGeometries(device, desc);
  std::vector<uint32_t> counts;
  for (const auto &g : desc.geometries)
    counts.push_back((g.index_count ? g.index_count : g.vertex_count) / 3);
  for (const auto &g : desc.aabbs)
    counts.push_back(g.count);
  auto object = std::unique_ptr<VulkanBottomLevelAccelerationStructureImpl>(
      new VulkanBottomLevelAccelerationStructureImpl(device, desc));
  VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  if (desc.allow_update)
    info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  if (desc.allow_compaction)
    info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  info.geometryCount = uint32_t(geometries.size());
  info.pGeometries = geometries.data();
  if (!object->Allocate(info, counts, desc.compacted_size))
    return nullptr;
  return object.release();
}

ITopLevelAccelerationStructure *VulkanTopLevelAccelerationStructureImpl::Create(
    IVulkanDevice *device,
    const TopLevelAccelerationStructureDesc &desc) {
  if (!device->GetPhysicalDevice().ray_tracing_pipeline_feature && !device->GetPhysicalDevice().ray_query_feature)
    return nullptr;
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  std::vector<VkAccelerationStructureGeometryKHR> geometries{geometry};
  std::vector<uint32_t> counts{desc.instance_count};
  auto object = std::unique_ptr<VulkanTopLevelAccelerationStructureImpl>(
      new VulkanTopLevelAccelerationStructureImpl(device, desc));
  VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  if (desc.allow_update)
    info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  if (desc.allow_compaction)
    info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  info.geometryCount = uint32_t(geometries.size());
  info.pGeometries = geometries.data();
  if (!object->Allocate(info, counts, desc.compacted_size))
    return nullptr;
  return object.release();
}

IBottomLevelAccelerationStructure *CreateVulkanBottomLevelAccelerationStructure(
    IVulkanDevice *device,
    const BottomLevelAccelerationStructureDesc &desc) {
  return VulkanBottomLevelAccelerationStructureImpl::Create(device, desc);
}

ITopLevelAccelerationStructure *IVulkanTopLevelAccelerationStructure::Create(
    IVulkanDevice *device,
    const TopLevelAccelerationStructureDesc &desc) {
  return VulkanTopLevelAccelerationStructureImpl::Create(device, desc);
}

VulkanBottomLevelAccelerationStructureImpl::~VulkanBottomLevelAccelerationStructureImpl() {
  DestroyVulkanAccelerationStructure();
}

VulkanTopLevelAccelerationStructureImpl::~VulkanTopLevelAccelerationStructureImpl() {
  // Retain referenced BLAS until the TLAS native storage has been destroyed.
  DestroyVulkanAccelerationStructure();
}
}  // namespace grassland::rhi::backend::vulkan
