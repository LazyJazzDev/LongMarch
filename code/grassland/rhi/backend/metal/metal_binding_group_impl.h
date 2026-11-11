#pragma once
#include "grassland/rhi/backend/backend_binding_group_bindinggroup_state.h"
#include "grassland/rhi/backend/backend_binding_group_bindingversion_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"

namespace grassland::rhi::backend::metal {
class MetalBindingVersionImpl final : public Object<IMetalBindingVersion> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IObject, IMetalBindingVersion, IBackendBindingVersion, IObject)
  static MetalBindingVersionImpl *Create(IMetalDevice *device, const BindingLayoutDesc &layout);
  ObjectPtr<IBackendBindingVersion> Clone() const override;
  bool IsInitialized(size_t binding, uint32_t element) const override;
  void Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const;
  void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const;

 private:
  IMetalDevice *device_;
  MetalBindingVersionImpl(IMetalDevice *device, const BindingLayoutDesc &layout);
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<BufferView> &values) override;
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<IImage *> &values) override;
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<ISampler *> &values) override;
  bool WriteDescriptors(size_t binding,
                        uint32_t first,
                        const std::vector<ITopLevelAccelerationStructure *> &values) override;
  std::vector<uint32_t> offsets_;
  std::vector<std::vector<bool>> initialized_;
  NS::SharedPtr<MTL::ArgumentEncoder> encoder_;
  NS::SharedPtr<MTL::Buffer> arguments_, sizes_;

 public:
  ~MetalBindingVersionImpl() override {
  }

 private:
  BindingVersionState state_BindingVersion_;

  BindingVersionState &StateBindingVersion() override {
    return state_BindingVersion_;
  }

  const BindingVersionState &StateBindingVersion() const override {
    return state_BindingVersion_;
  }
};

class MetalBindingGroupImpl final : public Object<IMetalBindingGroup> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBindingGroup, IMetalBindingGroup, IBackendBindingGroup, IBindingGroup)
  static IMetalBindingGroup *Create(IMetalDevice *device, const BindingLayoutDesc &layout);

 private:
  MetalBindingGroupImpl(IMetalDevice *device, ObjectPtr<Version> version);

 public:
  ~MetalBindingGroupImpl() override {
  }

 private:
  BindingGroupState state_BindingGroup_;

  BindingGroupState &StateBindingGroup() override {
    return state_BindingGroup_;
  }

  const BindingGroupState &StateBindingGroup() const override {
    return state_BindingGroup_;
  }

 public:
  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, structures, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, buffers, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, images, first_element);
  }

  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override {
    return IBackendBindingGroup::CommonBindResource(binding, samplers, first_element);
  }

  const BindingLayoutDesc &GetLayout() const override {
    return IBackendBindingGroup::CommonGetLayout();
  }
};
}  // namespace grassland::rhi::backend::metal
