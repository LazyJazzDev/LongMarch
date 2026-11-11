#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendBindingGroup)

namespace grassland::rhi::backend {
class IBackendBindingVersion : public IObject {
 public:
  virtual bool IsUniquelyOwned() const noexcept = 0;
  virtual const BindingLayoutDesc &GetLayout() const = 0;
  virtual const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const = 0;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  virtual bool IsComplete() const = 0;
  virtual ObjectPtr<IBackendBindingVersion> Clone() const = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) = 0;
  virtual bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) = 0;
  virtual bool Write(size_t binding_index,
                     uint32_t first_element,
                     const std::vector<ITopLevelAccelerationStructure *> &values) = 0;
};

class IBackendBindingGroup : public IObject {
 public:
  using Version = IBackendBindingVersion;
  virtual IBackendDevice *Owner() const = 0;
  virtual  // Borrowed until the next mutation; the context retains it before recording use.
      Version *
      CurrentVersion() const = 0;
};

struct MappedBinding {
  BindingDesc desc;
  uint32_t group{}, shader_register{}, table_offset{};
};

std::vector<MappedBinding> MapBindings(const PipelineLayoutDesc &layout, BackendAPI api);
bool ValidateBindingLayout(const BindingLayoutDesc &layout);
bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b);
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendBindingVersion)
