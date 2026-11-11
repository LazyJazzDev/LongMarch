#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
// Argument IDs are dense and independent of shader liveness and stage visibility.
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout);
bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout);

class IMetalBindingVersion : public IBackendBindingVersion {
 public:
  static IMetalBindingVersion *Create(IMetalDevice *device, const BindingLayoutDesc &layout);

  virtual bool IsInitialized(size_t binding, uint32_t element) const = 0;
  virtual void Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const = 0;
  virtual void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const = 0;
};

IBindingGroup *CreateMetalBindingGroup(IMetalDevice *device, const BindingLayoutDesc &layout);

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalBindingVersion,
                        grassland::rhi::backend::IBackendBindingVersion,
                        "fcef2422-11d0-4f7f-acb2-f3fcd629a898")
