#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendBindingGroup)
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend {
struct BindingVersionState;

class IBackendBindingVersion : public IObject {
 public:
  virtual bool IsUniquelyOwned() const noexcept = 0;

 protected:
  void InitializeBindingVersion(const BindingLayoutDesc &layout);

 public:
  const BindingLayoutDesc &GetLayout() const;
  const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  bool IsComplete() const;
  virtual ObjectPtr<IBackendBindingVersion> Clone() const = 0;
  bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values);
  bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values);
  bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values);
  bool Write(size_t binding_index, uint32_t first_element, const std::vector<ITopLevelAccelerationStructure *> &values);

 protected:
  virtual bool WriteDescriptors(size_t binding_index,
                                uint32_t first_element,
                                const std::vector<ITopLevelAccelerationStructure *> &values) = 0;
  bool CommonWriteDescriptors(size_t binding_index,
                              uint32_t first_element,
                              const std::vector<ITopLevelAccelerationStructure *> &values);
  virtual bool WriteDescriptors(size_t binding_index,
                                uint32_t first_element,
                                const std::vector<BufferView> &values) = 0;
  virtual bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) = 0;
  virtual bool WriteDescriptors(size_t binding_index,
                                uint32_t first_element,
                                const std::vector<ISampler *> &values) = 0;
  virtual BindingVersionState &StateBindingVersion() = 0;
  virtual const BindingVersionState &StateBindingVersion() const = 0;
};

struct BindingGroupState;

class IBackendBindingGroup : public IBindingGroup {
 public:
  using Version = IBackendBindingVersion;

 protected:
  void InitializeBindingGroup(IBackendDevice *device, ObjectPtr<Version> version);

 public:
  using IBindingGroup::BindResource;
  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override = 0;
  bool CommonBindResource(uint32_t binding,
                          const std::vector<ITopLevelAccelerationStructure *> &structures,
                          uint32_t first_element = 0);
  const BindingLayoutDesc &GetLayout() const override = 0;
  const BindingLayoutDesc &CommonGetLayout() const;
  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override = 0;
  bool CommonBindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0);
  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override = 0;
  bool CommonBindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0);
  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override = 0;
  bool CommonBindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0);
  IBackendDevice *Owner() const;
  // Borrowed until the next mutation; the context retains it before recording use.
  Version *CurrentVersion() const;

 private:
  const BindingDesc *FindBinding(uint32_t binding, uint32_t first_element, size_t count, size_t &binding_index) const;

 protected:
  virtual BindingGroupState &StateBindingGroup() = 0;
  virtual const BindingGroupState &StateBindingGroup() const = 0;
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
