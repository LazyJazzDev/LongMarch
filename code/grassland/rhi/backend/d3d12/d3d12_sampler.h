#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Sampler)
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Sampler : public IBackendSampler {
 public:
  static ID3D12Sampler *Create(ID3D12Device *device, const SamplerDesc &desc);
  virtual const D3D12_SAMPLER_DESC &NativeDesc() const = 0;
  virtual D3D12_CPU_DESCRIPTOR_HANDLE Descriptor() const = 0;
};

;
}  // namespace grassland::rhi::backend::d3d12
