#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;
// Argument IDs are dense and independent of shader liveness and stage visibility.
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout);
bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout);

class MetalBindingGroup final : public BindingGroup {
 public:
  class Version final : public BindingGroup::Version {
   public:
    static Version *Create(MetalDevice *device, const BindingLayoutDesc &layout);
    ObjectPtr<BindingGroup::Version> Clone() const override;
    bool IsInitialized(size_t binding, uint32_t element) const override;
    void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const;

   private:
    Version(MetalDevice *device, const BindingLayoutDesc &layout);
    bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<BufferView> &values) override;
    bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<IImage *> &values) override;
    bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<SamplerDesc> &values) override;
    std::vector<uint32_t> offsets_;
    std::vector<std::vector<bool>> initialized_;
    std::vector<std::vector<NS::SharedPtr<MTL::SamplerState>>> samplers_;
    NS::SharedPtr<MTL::ArgumentEncoder> encoder_;
    NS::SharedPtr<MTL::Buffer> arguments_, sizes_;
  };

  static MetalBindingGroup *Create(MetalDevice *device, const BindingLayoutDesc &layout);

 private:
  MetalBindingGroup(MetalDevice *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::metal
