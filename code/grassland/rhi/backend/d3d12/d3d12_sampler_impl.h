#pragma once
#include "grassland/rhi/backend/backend_sampler_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12SamplerImpl final : public Object<ID3D12Sampler> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ISampler, ID3D12Sampler, Sampler, ISampler)
  static ID3D12Sampler *Create(ID3D12Device *device, const SamplerDesc &desc);

  const D3D12_SAMPLER_DESC &NativeDesc() const {
    return native_desc_;
  }

  D3D12_CPU_DESCRIPTOR_HANDLE Descriptor() const {
    return descriptor_->CPU();
  }

 private:
  D3D12SamplerImpl(ID3D12Device *device, const SamplerDesc &desc);
  D3D12_SAMPLER_DESC native_desc_{};
  ObjectPtr<D3D12DescriptorPool::Lease> descriptor_;

 public:
  ~D3D12SamplerImpl() override {
  }

 private:
  SamplerState state_Sampler_;

  SamplerState &StateSampler() override {
    return state_Sampler_;
  }

  const SamplerState &StateSampler() const override {
    return state_Sampler_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
