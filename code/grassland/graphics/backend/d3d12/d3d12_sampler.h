#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Sampler : public Sampler {
 public:
  D3D12Sampler(D3D12Core *core, const SamplerInfo &info);
  const D3D12_SAMPLER_DESC &SamplerDesc() const {
    return sampler_desc_;
  }

 private:
  D3D12Core *core_;
  D3D12_SAMPLER_DESC sampler_desc_;
};

}  // namespace grassland::graphics::backend
