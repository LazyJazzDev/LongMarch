#include "grassland/d3d12/device.h"

#include "grassland/d3d12/buffer.h"
#include "grassland/d3d12/command_allocator.h"
#include "grassland/d3d12/command_list.h"
#include "grassland/d3d12/command_queue.h"
#include "grassland/d3d12/descriptor_heap.h"
#include "grassland/d3d12/fence.h"
#include "grassland/d3d12/image.h"
#include "grassland/d3d12/root_signature.h"
#include "grassland/d3d12/shader_module.h"

namespace grassland::d3d12 {

Device::Device(const class Adapter &adapter,
               const D3D_FEATURE_LEVEL feature_level,
               ComPtr<ID3D12Device> device)
    : adapter_(adapter),
      feature_level_(feature_level),
      device_(std::move(device)) {
}

HRESULT Device::CreateCommandQueue(D3D12_COMMAND_LIST_TYPE type,
                                   double_ptr<CommandQueue> pp_command_queue) {
  D3D12_COMMAND_QUEUE_DESC desc = {};
  desc.Type = type;

  ComPtr<ID3D12CommandQueue> command_queue;
  RETURN_IF_FAILED_HR(
      device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&command_queue)),
      "failed to create command queue.");

  pp_command_queue.construct(command_queue);
  return S_OK;
}

HRESULT Device::CreateCommandAllocator(
    D3D12_COMMAND_LIST_TYPE type,
    double_ptr<CommandAllocator> pp_command_allocator) {
  ComPtr<ID3D12CommandAllocator> command_allocator;
  RETURN_IF_FAILED_HR(
      device_->CreateCommandAllocator(type, IID_PPV_ARGS(&command_allocator)),
      "failed to create command allocator.");

  pp_command_allocator.construct(command_allocator);
  return S_OK;
}

HRESULT Device::CreateDescriptorHeap(
    const D3D12_DESCRIPTOR_HEAP_DESC &desc,
    double_ptr<DescriptorHeap> pp_descriptor_heap) {
  ComPtr<ID3D12DescriptorHeap> descriptor_heap;
  RETURN_IF_FAILED_HR(
      device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&descriptor_heap)),
      "failed to create descriptor heap.");
  pp_descriptor_heap.construct(descriptor_heap);
  return S_OK;
}

HRESULT Device::CreateDescriptorHeap(
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    uint32_t num_descriptors,
    double_ptr<DescriptorHeap> pp_descriptor_heap) {
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Type = type;
  desc.NumDescriptors = num_descriptors;
  desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) {
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  }

  return CreateDescriptorHeap(desc, pp_descriptor_heap);
}

HRESULT Device::CreateFence(double_ptr<Fence> pp_fence) {
  ComPtr<ID3D12Fence> fence;

  device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

  pp_fence.construct(fence);

  return S_OK;
}

}  // namespace grassland::d3d12
