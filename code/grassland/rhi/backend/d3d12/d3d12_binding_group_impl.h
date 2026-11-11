#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12BindingVersionImpl final : public Object<ID3D12BindingVersion> {
 public:
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
                        const std::vector<ITopLevelAccelerationStructure *> &values);
  D3D12BindingVersionImpl(ID3D12Device *device, const BindingLayoutDesc &layout);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values);
  bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values);
  std::vector<uint32_t> offsets_;
  std::vector<std::vector<bool>> initialized_;
  ObjectPtr<D3D12DescriptorPool::Lease> resources_, samplers_;

 public:
  ~D3D12BindingVersionImpl() override = default;

 private:
  void InitializeBindingVersion(const BindingLayoutDesc &layout);

 public:
  const BindingLayoutDesc &GetLayout() const override;

  const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const override;

  bool IsComplete() const override;

  bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;

  bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;

  bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) override;

  bool Write(size_t binding_index,
             uint32_t first_element,
             const std::vector<ITopLevelAccelerationStructure *> &values) override;

 private:
  BindingLayoutDesc layout_;
  // Indexed by sorted layout binding, then array element.
  std::vector<std::vector<ObjectPtr<IObject>>> references_;
};

class D3D12BindingGroupImpl final : public Object<rhi::IBindingGroup, IBackendBindingGroup> {
 public:
  using Version = ID3D12BindingVersion;
  static IBindingGroup *Create(ID3D12Device *device, const BindingLayoutDesc &layout);

 private:
  D3D12BindingGroupImpl(ID3D12Device *device, ObjectPtr<Version> version);

 public:
  ~D3D12BindingGroupImpl() override = default;

 private:
  void InitializeBindingGroup(IBackendDevice *device, ObjectPtr<IBackendBindingVersion> version);

 public:
  const BindingLayoutDesc &GetLayout() const override;

  IBackendDevice *Owner() const override;

  // Borrowed until the next mutation; the context retains it before recording use.
  IBackendBindingVersion *CurrentVersion() const override;

 private:
  const BindingDesc *FindBinding(uint32_t binding, uint32_t first_element, size_t count, size_t &binding_index) const;

 public:
  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override;

  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override;

  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override;

  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override;

 private:
  IBackendDevice *device_;
  BindingLayoutDesc layout_;
  ObjectPtr<IBackendBindingVersion> version_;
};
}  // namespace grassland::rhi::backend::d3d12
