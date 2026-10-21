#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12Sampler final : public Sampler {
 public:
  static D3D12Sampler *Create(D3D12Device *device, const SamplerDesc &desc);

  const D3D12_SAMPLER_DESC &NativeDesc() const {
    return native_desc_;
  }

  D3D12_CPU_DESCRIPTOR_HANDLE Descriptor() const {
    return descriptor_->CPU();
  }

 private:
  D3D12Sampler(D3D12Device *device, const SamplerDesc &desc);
  D3D12_SAMPLER_DESC native_desc_{};
  ObjectPtr<D3D12DescriptorPool::Lease> descriptor_;
};
}  // namespace grassland::rhi::backend::d3d12
