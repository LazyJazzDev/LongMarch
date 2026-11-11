#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalBindingGroup)
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::metal {
// Argument IDs are dense and independent of shader liveness and stage visibility.
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout);
bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout);

class IMetalBindingVersion : public IBackendBindingVersion {
 public:
  static IMetalBindingVersion *Create(IMetalDevice *device, const BindingLayoutDesc &layout);
  virtual ObjectPtr<IBackendBindingVersion> Clone() const = 0;
  virtual bool IsInitialized(size_t binding, uint32_t element) const = 0;
  virtual void Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const = 0;
  virtual void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const = 0;
};

;

class IMetalBindingGroup : public IBackendBindingGroup {
 public:
  using Version = IMetalBindingVersion;
  static IMetalBindingGroup *Create(IMetalDevice *device, const BindingLayoutDesc &layout);
};

;
}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalBindingVersion)
