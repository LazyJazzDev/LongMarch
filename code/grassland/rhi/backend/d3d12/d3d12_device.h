#pragma once
#include <d3d12sdklayers.h>

#include <tuple>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_scratch_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_transient_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Device)

namespace grassland::rhi::backend::d3d12 {
class ID3D12Device : public IBackendDevice {
 public:
  static rhi::IDevice *Create(PhysicalDevice info, ICore *core, ComPtr<::ID3D12Device> device);
  virtual ::ID3D12Device *NativeDevice() const = 0;
  virtual D3D12MA::Allocator *Allocator() const = 0;
  virtual D3D12DescriptorPool &RTVPool() = 0;
  virtual D3D12DescriptorPool &DSVPool() = 0;
  virtual D3D12TransientDescriptorPool &ResourceDescriptors() = 0;
  virtual D3D12TransientDescriptorPool &SamplerDescriptors() = 0;
  virtual D3D12DescriptorPool &ResourceDescriptorStorage() = 0;
  virtual D3D12DescriptorPool &SamplerDescriptorStorage() = 0;
  virtual IDXGIFactory4 *Factory() const = 0;
  virtual bool SupportsTearing() const = 0;
  virtual D3D12TemporalBufferPool &UploadPool() = 0;
  virtual D3D12TemporalBufferPool &DownloadPool() = 0;
  virtual D3D12ScratchPool &ScratchPool() = 0;
  virtual  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
      // The pool grows on demand, independently of swapchain image count.
      bool
      AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator, ComPtr<ID3D12GraphicsCommandList> &list) = 0;
  virtual void RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator,
                                       ComPtr<ID3D12GraphicsCommandList> list) = 0;
};

}  // namespace grassland::rhi::backend::d3d12
