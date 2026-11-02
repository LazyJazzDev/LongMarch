#include "grassland/rhi/backend/d3d12/d3d12_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12CUDABuffer::D3D12CUDABuffer(D3D12Device *device, const BufferDesc &desc)
    : D3D12Buffer(device, desc),
      mapping_(device->GetPhysicalDevice().cuda_device) {
}

ID3D12Resource *D3D12CUDABuffer::NativeBuffer() const {
  return resource_.Get();
}

D3D12_RESOURCE_STATES D3D12CUDABuffer::DefaultState() const {
  return D3D12_RESOURCE_STATE_COMMON;
}

void *D3D12CUDABuffer::GetDevicePointer() const {
  return mapping_.Pointer();
}

D3D12CUDABuffer *D3D12CUDABuffer::Create(D3D12Device *device, const BufferDesc &desc) {
  if (!desc.size || device->GetPhysicalDevice().cuda_device < 0)
    return nullptr;
  auto object = std::unique_ptr<D3D12CUDABuffer>(new D3D12CUDABuffer(device, desc));
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap.CreationNodeMask = heap.VisibleNodeMask = 1;
  D3D12_RESOURCE_DESC resource{};
  resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resource.Width = desc.size;
  resource.Height = resource.DepthOrArraySize = resource.MipLevels = 1;
  resource.SampleDesc.Count = 1;
  resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if (!Check(device->NativeDevice()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &resource,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&object->resource_)),
             "Create shared CUDA buffer"))
    return nullptr;
  HANDLE handle{};
  if (!Check(
          device->NativeDevice()->CreateSharedHandle(object->resource_.Get(), nullptr, GENERIC_ALL, nullptr, &handle),
          "Export CUDA buffer"))
    return nullptr;
  cudaExternalMemoryHandleDesc memory{};
  memory.type = cudaExternalMemoryHandleTypeD3D12Resource;
  memory.handle.win32.handle = handle;
  memory.size = device->NativeDevice()->GetResourceAllocationInfo(0, 1, &resource).SizeInBytes;
  memory.flags = cudaExternalMemoryDedicated;
  const bool imported = object->mapping_.Import(memory, desc.size);
  CloseHandle(handle);
  return imported ? object.release() : nullptr;
}

D3D12CUDAInteropTimeline::D3D12CUDAInteropTimeline(D3D12Device *device) : CUDAInteropTimeline(device) {
}

D3D12CUDAInteropTimeline::~D3D12CUDAInteropTimeline() {
  Reset();
}

ID3D12Fence *D3D12CUDAInteropTimeline::NativeFence() const {
  return fence_.Get();
}

D3D12CUDAInteropTimeline *D3D12CUDAInteropTimeline::Create(D3D12Device *device) {
  if (device->GetPhysicalDevice().cuda_device < 0)
    return nullptr;
  auto object = std::unique_ptr<D3D12CUDAInteropTimeline>(new D3D12CUDAInteropTimeline(device));
  if (!Check(device->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&object->fence_)),
             "Create CUDA shared fence"))
    return nullptr;
  HANDLE handle{};
  if (!Check(device->NativeDevice()->CreateSharedHandle(object->fence_.Get(), nullptr, GENERIC_ALL, nullptr, &handle),
             "Export CUDA fence"))
    return nullptr;
  cudaExternalSemaphoreHandleDesc desc{};
  desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
  desc.handle.win32.handle = handle;
  const bool imported = object->Import(desc);
  CloseHandle(handle);
  return imported ? object.release() : nullptr;
}

ICUDABuffer *D3D12Device::CreateCUDABuffer(const BufferDesc &desc) {
  return D3D12CUDABuffer::Create(this, desc);
}

ICUDAInteropTimeline *D3D12Device::CreateCUDAInteropTimeline() {
  return D3D12CUDAInteropTimeline::Create(this);
}
}  // namespace grassland::rhi::backend::d3d12
#endif
