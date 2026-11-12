#include "grassland/rhi/backend/d3d12/d3d12_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12CUDABufferImpl::D3D12CUDABufferImpl(ID3D12Device *device, const BufferDesc &desc, int cuda_device)
    : owner_(device),
      desc_(desc),
      mapping_(cuda_device) {
}

BufferDesc D3D12CUDABufferImpl::GetDesc() const {
  return desc_;
}

ID3D12Resource *D3D12CUDABufferImpl::NativeBuffer() const {
  return resource_.Get();
}

D3D12_RESOURCE_STATES D3D12CUDABufferImpl::DefaultState() const {
  return D3D12_RESOURCE_STATE_COMMON;
}

void *D3D12CUDABufferImpl::GetDevicePointer() const {
  return mapping_.Pointer();
}

rhi::ICUDABuffer *D3D12CUDABufferImpl::Create(ID3D12Device *device, const BufferDesc &desc) {
  if (!desc.size)
    return nullptr;
  const int ordinal = device->GetPhysicalDevice().cuda_device;
  if (ordinal < 0) {
    ReportError(-1, "No CUDA device matches this D3D12 adapter.");
    return nullptr;
  }
  auto object = std::unique_ptr<D3D12CUDABufferImpl>(new D3D12CUDABufferImpl(device, desc, ordinal));
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

}  // namespace grassland::rhi::backend::d3d12
#endif
