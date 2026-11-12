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
