#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12SamplerImpl final : public Object<ID3D12Sampler> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ISampler, ID3D12Sampler, IBackendSampler, ISampler)

  static ID3D12Sampler *Create(ID3D12Device *device, const SamplerDesc &desc);

  const D3D12_SAMPLER_DESC &NativeDesc() const {
    return native_desc_;
  }

  D3D12_CPU_DESCRIPTOR_HANDLE Descriptor() const {
    return descriptor_->CPU();
  }

  ~D3D12SamplerImpl() override = default;

  const SamplerDesc &GetDesc() const override {
    return desc_;
  }

  const void *DeviceIdentity() const override {
    return device_identity_;
  }

 private:
  D3D12SamplerImpl(ID3D12Device *device, const SamplerDesc &desc);

  D3D12_SAMPLER_DESC native_desc_{};

  ObjectPtr<D3D12DescriptorPool::Lease> descriptor_;

  void InitializeSampler(const SamplerDesc &desc, const void *device) {
    desc_ = desc;
    device_identity_ = device;
  }

  SamplerDesc desc_{};

  const void *device_identity_{};
};
}  // namespace grassland::rhi::backend::d3d12
