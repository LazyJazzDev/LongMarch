#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"

#include <cstring>

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#define D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
#include <D3D12MemAlloc.h>
#undef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED

namespace grassland::rhi::backend::d3d12 {
D3D12AccelerationStructure::D3D12AccelerationStructure(D3D12Device *device) : device_(device) {
}

D3D12Device *D3D12AccelerationStructure::Owner() const {
  return device_;
}

uint64_t D3D12AccelerationStructure::ScratchSize(AccelerationStructureBuildMode mode) const {
  return mode == ACCELERATION_STRUCTURE_UPDATE ? update_scratch_ : build_scratch_;
}

bool D3D12AccelerationStructure::AllowsUpdate() const {
  return allow_update_;
}

bool D3D12AccelerationStructure::AllowsCompaction() const {
  return allow_compaction_;
}

bool D3D12AccelerationStructure::IsCompactStorage() const {
  return compact_;
}

D3D12BottomLevelAccelerationStructure::D3D12BottomLevelAccelerationStructure(
    D3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc)
    : D3D12AccelerationStructure(device),
      desc_(desc) {
}

const BottomLevelAccelerationStructureDesc &D3D12BottomLevelAccelerationStructure::GetDesc() const {
  return desc_;
}

AccelerationStructureType D3D12BottomLevelAccelerationStructure::GetType() const {
  return ACCELERATION_STRUCTURE_BOTTOM_LEVEL;
}

uint64_t D3D12BottomLevelAccelerationStructure::GetCompactedSize() const {
  return ReadCompactedSize();
}

D3D12TopLevelAccelerationStructure::D3D12TopLevelAccelerationStructure(D3D12Device *device,
                                                                       const TopLevelAccelerationStructureDesc &desc)
    : D3D12AccelerationStructure(device),
      desc_(desc) {
}

const TopLevelAccelerationStructureDesc &D3D12TopLevelAccelerationStructure::GetDesc() const {
  return desc_;
}

AccelerationStructureType D3D12TopLevelAccelerationStructure::GetType() const {
  return ACCELERATION_STRUCTURE_TOP_LEVEL;
}

uint64_t D3D12TopLevelAccelerationStructure::GetCompactedSize() const {
  return ReadCompactedSize();
}

const std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> &D3D12TopLevelAccelerationStructure::BottomLevels()
    const {
  return bottom_levels_;
}

void D3D12TopLevelAccelerationStructure::CommitDependencies(
    std::vector<ObjectPtr<IBottomLevelAccelerationStructure>> references) {
  bottom_levels_ = std::move(references);
}

D3D12AccelerationStructure::~D3D12AccelerationStructure() {
  resource_.Reset();
  query_buffer_.Reset();
  query_output_.Reset();
}

ID3D12Resource *D3D12AccelerationStructure::NativeResource() const {
  return resource_.Get();
}

ID3D12Resource *D3D12AccelerationStructure::CompactionReadback() const {
  return query_buffer_.Get();
}

ID3D12Resource *D3D12AccelerationStructure::CompactionOutput() const {
  return query_output_.Get();
}

D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS D3D12AccelerationStructure::NativeInputs() const {
  return inputs_;
}

uint64_t D3D12AccelerationStructure::Address() const {
  return resource_->GetGPUVirtualAddress();
}

uint64_t D3D12AccelerationStructure::ReadCompactedSize() const {
  if (!query_buffer_)
    return 0;
  void *data{};
  D3D12_RANGE range{0, sizeof(uint64_t)};
  if (!Check(query_buffer_->Map(0, &range, &data), "Map compacted size"))
    return 0;
  uint64_t result{};
  std::memcpy(&result, data, sizeof(result));
  D3D12_RANGE written{0, 0};
  query_buffer_->Unmap(0, &written);
  return result;
}

bool D3D12AccelerationStructure::Allocate(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &inputs,
                                          uint64_t compacted_size) {
  ComPtr<ID3D12Device5> device;
  if (!Check(device_->NativeDevice()->QueryInterface(IID_PPV_ARGS(&device)), "Query ray tracing device"))
    return false;
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes{};
  device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &sizes);
  if (!sizes.ResultDataMaxSizeInBytes)
    return false;
  inputs_ = inputs;
  inputs_.pGeometryDescs = nullptr;
  allow_update_ = (inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) != 0;
  allow_compaction_ = (inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0;
  compact_ = compacted_size != 0;
  if (compact_ && !allow_compaction_)
    return false;
  build_scratch_ = sizes.ScratchDataSizeInBytes;
  update_scratch_ = sizes.UpdateScratchDataSizeInBytes;
  D3D12_RESOURCE_DESC info{};
  info.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  info.Width = compacted_size ? compacted_size : sizes.ResultDataMaxSizeInBytes;
  constexpr uint64_t alignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
  if (info.Width > UINT64_MAX - alignment + 1)
    return false;
  info.Width = (info.Width + alignment - 1) & ~(alignment - 1);
  info.Height = info.DepthOrArraySize = info.MipLevels = info.SampleDesc.Count = 1;
  info.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  info.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  D3D12MA::ALLOCATION_DESC memory{};
  memory.HeapType = D3D12_HEAP_TYPE_DEFAULT;
  if (!Check(
          device_->Allocator()->CreateResource(&memory, &info, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                               nullptr, &allocation_, IID_PPV_ARGS(&resource_)),
          "Allocate acceleration structure"))
    return false;
  if (allow_compaction_) {
    info.Width = sizeof(uint64_t);
    if (!Check(device_->Allocator()->CreateResource(&memory, &info, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                    &query_output_allocation_, IID_PPV_ARGS(&query_output_)),
               "Allocate compacted size output"))
      return false;
    memory.HeapType = D3D12_HEAP_TYPE_READBACK;
    info.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (!Check(device_->Allocator()->CreateResource(&memory, &info, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    &query_allocation_, IID_PPV_ARGS(&query_buffer_)),
               "Allocate compacted size readback"))
      return false;
  }
  return true;
}

std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> NativeGeometries(
    D3D12Device *,
    const BottomLevelAccelerationStructureDesc &desc,
    const std::vector<AccelerationStructureGeometryBuffers> &buffers) {
  std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> result;
  for (size_t i = 0; i < desc.geometries.size(); ++i) {
    const auto &g = desc.geometries[i];
    D3D12_RAYTRACING_GEOMETRY_DESC native{};
    native.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    native.Flags = g.opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    auto &tri = native.Triangles;
    tri.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    tri.VertexCount = g.vertex_count;
    tri.VertexBuffer.StrideInBytes = g.vertex_stride;
    tri.IndexCount = g.index_count;
    tri.IndexFormat = !g.index_count                          ? DXGI_FORMAT_UNKNOWN
                      : g.index_format == INDEX_FORMAT_UINT16 ? DXGI_FORMAT_R16_UINT
                                                              : DXGI_FORMAT_R32_UINT;
    if (!buffers.empty()) {
      tri.VertexBuffer.StartAddress =
          static_cast<D3D12Buffer *>(buffers[i].data.buffer)->NativeBuffer()->GetGPUVirtualAddress() +
          buffers[i].data.offset;
      if (g.index_count)
        tri.IndexBuffer =
            static_cast<D3D12Buffer *>(buffers[i].indices.buffer)->NativeBuffer()->GetGPUVirtualAddress() +
            buffers[i].indices.offset;
    }
    result.push_back(native);
  }
  for (size_t i = 0; i < desc.aabbs.size(); ++i) {
    const auto &g = desc.aabbs[i];
    D3D12_RAYTRACING_GEOMETRY_DESC native{};
    native.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    native.Flags = g.opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    native.AABBs.AABBCount = g.count;
    native.AABBs.AABBs.StrideInBytes = g.stride;
    if (!buffers.empty())
      native.AABBs.AABBs.StartAddress =
          static_cast<D3D12Buffer *>(buffers[i].data.buffer)->NativeBuffer()->GetGPUVirtualAddress() +
          buffers[i].data.offset;
    result.push_back(native);
  }
  return result;
}

D3D12BottomLevelAccelerationStructure *D3D12BottomLevelAccelerationStructure::Create(
    D3D12Device *device,
    const BottomLevelAccelerationStructureDesc &desc) {
  if (!device->SupportRayTracingPipeline() && !device->SupportRayQuery())
    return nullptr;
  if (!ValidateBottomLevelDesc(desc))
    return nullptr;
  auto geometries = NativeGeometries(device, desc);
  auto object =
      std::unique_ptr<D3D12BottomLevelAccelerationStructure>(new D3D12BottomLevelAccelerationStructure(device, desc));
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
  inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
  inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
  if (desc.allow_update)
    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
  if (desc.allow_compaction)
    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
  inputs.NumDescs = UINT(geometries.size());
  inputs.pGeometryDescs = geometries.data();
  if (!object->Allocate(inputs, desc.compacted_size))
    return nullptr;
  return object.release();
}

D3D12TopLevelAccelerationStructure *D3D12TopLevelAccelerationStructure::Create(
    D3D12Device *device,
    const TopLevelAccelerationStructureDesc &desc) {
  if (!device->SupportRayTracingPipeline() && !device->SupportRayQuery())
    return nullptr;
  auto object =
      std::unique_ptr<D3D12TopLevelAccelerationStructure>(new D3D12TopLevelAccelerationStructure(device, desc));
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
  inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
  if (desc.allow_update)
    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
  if (desc.allow_compaction)
    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
  inputs.NumDescs = desc.instance_count;
  if (!object->Allocate(inputs, desc.compacted_size))
    return nullptr;
  return object.release();
}
}  // namespace grassland::rhi::backend::d3d12
