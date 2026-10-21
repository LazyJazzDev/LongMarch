#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend {
class Device;

class BindingGroup : public Object<IBindingGroup> {
 public:
  class Version : public Object<IObject> {
   public:
    Version(Device *device, const BindingLayoutDesc &layout);
    const BindingLayoutDesc &GetLayout() const;
    const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const;
    virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
    bool IsComplete() const;
    virtual ObjectPtr<Version> Clone() const = 0;
    bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values);
    bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values);
    bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values);

   protected:
    virtual bool WriteDescriptors(size_t binding_index,
                                  uint32_t first_element,
                                  const std::vector<BufferView> &values) = 0;
    virtual bool WriteDescriptors(size_t binding_index,
                                  uint32_t first_element,
                                  const std::vector<IImage *> &values) = 0;
    virtual bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values);
    // Internal bridge for the unchanged Metal implementation.
    virtual bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<SamplerDesc> &values);
    Device *device_;
    BindingLayoutDesc layout_;
    // Indexed by sorted layout binding, then array element.
    // Native sampler backends populate sampler slots; the legacy adapter leaves them empty.
    std::vector<std::vector<ObjectPtr<IObject>>> references_;
  };

  BindingGroup(Device *device, ObjectPtr<Version> version);
  using IBindingGroup::BindResource;
  const BindingLayoutDesc &GetLayout() const override;
  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override;
  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override;
  bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) override;
  Device *Owner() const;
  // Borrowed until the next mutation; the context retains it before recording use.
  Version *CurrentVersion() const;

 private:
  const BindingDesc *FindBinding(uint32_t binding, uint32_t first_element, size_t count, size_t &binding_index) const;
  Device *device_;
  BindingLayoutDesc layout_;
  ObjectPtr<Version> version_;
};

struct MappedBinding {
  BindingDesc desc;
  uint32_t group{}, shader_register{}, table_offset{};
};

std::vector<MappedBinding> MapBindings(const PipelineLayoutDesc &layout, BackendAPI api);
bool ValidateBindingLayout(const BindingLayoutDesc &layout);
bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b);
}  // namespace grassland::rhi::backend
