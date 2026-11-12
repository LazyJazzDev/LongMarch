#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12BindingVersion : public IObject {
 public:
  virtual bool IsUniquelyOwned() const noexcept = 0;
  virtual const BindingLayoutDesc &GetLayout() const = 0;
  virtual const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const = 0;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  virtual bool IsComplete() const = 0;
  virtual ObjectPtr<ID3D12BindingVersion> Clone() const = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) = 0;
  virtual bool Write(size_t binding_index,
                     uint32_t first_element,
                     const std::vector<ITopLevelAccelerationStructure *> &values) = 0;
  static ID3D12BindingVersion *Create(ID3D12Device *device, const BindingLayoutDesc &layout);

  virtual uint32_t ResourceCount() const = 0;
  virtual D3D12DescriptorPool::Lease *ResourceDescriptors() const = 0;
  virtual D3D12DescriptorPool::Lease *SamplerDescriptors() const = 0;
};

class ID3D12BindingGroup : public IObject {
 public:
  virtual ID3D12Device *Owner() const = 0;
  // Borrowed until the next mutation; retain before recording use.
  virtual ID3D12BindingVersion *CurrentVersion() const = 0;
};

IBindingGroup *CreateD3D12BindingGroup(ID3D12Device *device, const BindingLayoutDesc &layout);

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12BindingVersion,
                        grassland::rhi::IObject,
                        "1e82b239-9a3f-4e32-aeab-b2fdaba168ff")

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12BindingGroup,
                        grassland::rhi::IObject,
                        "6fe24e6a-3907-4958-b6e3-499aab48e4e4")

namespace grassland::rhi::backend::d3d12 {
class D3D12BindingVersionImpl final : public Object<ID3D12BindingVersion> {
 public:
  static D3D12BindingVersionImpl *Create(ID3D12Device *device, const BindingLayoutDesc &layout);
  ObjectPtr<ID3D12BindingVersion> Clone() const override;
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

class D3D12BindingGroupImpl final : public Object<rhi::IBindingGroup, ID3D12BindingGroup> {
 public:
  using Version = ID3D12BindingVersion;
  static IBindingGroup *Create(ID3D12Device *device, const BindingLayoutDesc &layout);

 private:
  D3D12BindingGroupImpl(ID3D12Device *device, ObjectPtr<Version> version);

 public:
  ~D3D12BindingGroupImpl() override = default;

 private:
  void InitializeBindingGroup(ID3D12Device *device, ObjectPtr<ID3D12BindingVersion> version);

 public:
  const BindingLayoutDesc &GetLayout() const override;

  ID3D12Device *Owner() const override;

  // Borrowed until the next mutation; the context retains it before recording use.
  ID3D12BindingVersion *CurrentVersion() const override;

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
  ID3D12Device *device_;
  BindingLayoutDesc layout_;
  ObjectPtr<ID3D12BindingVersion> version_;
};
}  // namespace grassland::rhi::backend::d3d12
