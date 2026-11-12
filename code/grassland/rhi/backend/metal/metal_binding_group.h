#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
// Argument IDs are dense and independent of shader liveness and stage visibility.
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout);
bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout);

class IMetalBindingVersion : public IObject {
 public:
  virtual bool IsUniquelyOwned() const noexcept = 0;
  virtual const BindingLayoutDesc &GetLayout() const = 0;
  virtual const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const = 0;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  virtual bool IsComplete() const = 0;
  virtual ObjectPtr<IMetalBindingVersion> Clone() const = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) = 0;
  virtual bool Write(size_t binding_index,
                     uint32_t first_element,
                     const std::vector<ITopLevelAccelerationStructure *> &values) = 0;
  static IMetalBindingVersion *Create(IMetalDevice *device, const BindingLayoutDesc &layout);

  virtual bool IsInitialized(size_t binding, uint32_t element) const = 0;
  virtual void Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const = 0;
  virtual void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const = 0;
};

class IMetalBindingGroup : public IObject {
 public:
  virtual IMetalDevice *Owner() const = 0;
  // Borrowed until the next mutation; retain before recording use.
  virtual IMetalBindingVersion *CurrentVersion() const = 0;
};

IBindingGroup *CreateMetalBindingGroup(IMetalDevice *device, const BindingLayoutDesc &layout);

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalBindingVersion,
                        grassland::rhi::IObject,
                        "fcef2422-11d0-4f7f-acb2-f3fcd629a898")

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalBindingGroup,
                        grassland::rhi::IObject,
                        "06c9c9ec-349e-4bdd-882a-11335b44c88d")
