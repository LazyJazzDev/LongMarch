#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalBindingGroup;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalBindingGroup)
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;
// Argument IDs are dense and independent of shader liveness and stage visibility.
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout);
bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout);

class MetalBindingGroup final : public BindingGroup {
 public:
  LONGMARCH_RHI_QUERY(MetalBindingGroup, BindingGroup)

  class Version final : public BindingGroup::Version {
   public:
    using BindingGroup::Version::QueryInterface;
    bool QueryInterface(InterfaceID id, void **output) noexcept override;

    static Version *Create(MetalDevice *device, const BindingLayoutDesc &layout);
    ObjectPtr<BindingGroup::Version> Clone() const override;
    bool IsInitialized(size_t binding, uint32_t element) const override;
    void Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const;
    void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const;

   private:
    MetalDevice *device_;
    Version(MetalDevice *device, const BindingLayoutDesc &layout);
    bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<BufferView> &values) override;
    bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<IImage *> &values) override;
    bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<ISampler *> &values) override;
    bool WriteDescriptors(size_t binding,
                          uint32_t first,
                          const std::vector<ITopLevelAccelerationStructure *> &values) override;
    std::vector<uint32_t> offsets_;
    std::vector<std::vector<bool>> initialized_;
    NS::SharedPtr<MTL::ArgumentEncoder> encoder_;
    NS::SharedPtr<MTL::IBackendBuffer> arguments_, sizes_;
  };

  static MetalBindingGroup *Create(MetalDevice *device, const BindingLayoutDesc &layout);

 private:
  MetalBindingGroup(MetalDevice *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalBindingGroup::Version)

inline bool grassland::rhi::backend::metal::MetalBindingGroup::Version::QueryInterface(InterfaceID id,
                                                                                       void **output) noexcept {
  if (!output)
    return false;
  if (auto result = MatchInterface<Version>(id, this)) {
    *output = result;
    AddRef();
    return true;
  }
  return BindingGroup::Version::QueryInterface(id, output);
}
