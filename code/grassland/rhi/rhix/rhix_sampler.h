#pragma once
#include "grassland/rhi/rhi_sampler.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Sampler : public ObjectContainer {
 public:
  Sampler() noexcept = default;

  Sampler(rhi::ISampler *object) noexcept : ObjectContainer(object) {
  }

  rhi::ISampler *Get() const noexcept {
    return static_cast<rhi::ISampler *>(GetObject());
  }

  const rhi::SamplerDesc &GetDesc() const {
    return Get()->GetDesc();
  }
};
}  // namespace grassland::rhix
