#pragma once
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12Sampler,
                        grassland::rhi::IObject,
                        "d1e384fa-d35c-42e0-8d5f-110c64fe99d0")

namespace grassland::rhi::backend::d3d12 {
class ID3D12Sampler : public IObject {
 public:
  virtual const void *DeviceIdentity() const = 0;
  static rhi::ISampler *Create(ID3D12Device *device, const SamplerDesc &desc);
  virtual D3D12_CPU_DESCRIPTOR_HANDLE Descriptor() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
class D3D12SamplerImpl final : public Object<rhi::ISampler, ID3D12Sampler> {
 public:
  static rhi::ISampler *Create(ID3D12Device *device, const SamplerDesc &desc);

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
