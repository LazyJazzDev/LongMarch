#pragma once
#include "grassland/rhi/backend/backend_binding_group_bindinggroup_state.h"
#include "grassland/rhi/backend/backend_binding_group_bindingversion_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12BindingVersionImpl final : public Object<ID3D12BindingVersion> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IObject, ID3D12BindingVersion, IBackendBindingVersion, IObject)
  static D3D12BindingVersionImpl *Create(ID3D12Device *device, const BindingLayoutDesc &layout);
  ObjectPtr<IBackendBindingVersion> Clone() const override;
  bool IsInitialized(size_t binding_index, uint32_t element) const override;
  uint32_t ResourceCount() const;
  D3D12DescriptorPool::Lease *ResourceDescriptors() const;
  D3D12DescriptorPool::Lease *SamplerDescriptors() const;

 private:
  ID3D12Device *device_;
  bool WriteDescriptors(size_t binding_index,
                        uint32_t first_element,
                        const std::vector<ITopLevelAccelerationStructure *> &values) override;
  D3D12BindingVersionImpl(ID3D12Device *device, const BindingLayoutDesc &layout);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) override;
  std::vector<uint32_t> offsets_;
  std::vector<std::vector<bool>> initialized_;
  ObjectPtr<D3D12DescriptorPool::Lease> resources_, samplers_;

 public:
  ~D3D12BindingVersionImpl() override {
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

class D3D12BindingGroupImpl final : public Object<ID3D12BindingGroup> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IBindingGroup, ID3D12BindingGroup, IBackendBindingGroup, IBindingGroup)
  static ID3D12BindingGroup *Create(ID3D12Device *device, const BindingLayoutDesc &layout);

 private:
  D3D12BindingGroupImpl(ID3D12Device *device, ObjectPtr<Version> version);

 public:
  ~D3D12BindingGroupImpl() override {
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
}  // namespace grassland::rhi::backend::d3d12
