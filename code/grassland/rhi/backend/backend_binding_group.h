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
    const BufferView &GetBuffer(size_t slot) const;
    IImage *GetImage(size_t slot) const;
    const SamplerDesc &GetSampler(size_t slot) const;
    const std::vector<IImage *> &GetImages() const;
    bool IsInitialized(size_t slot) const;
    bool IsComplete() const;
    ObjectPtr<Version> Clone() const;
    bool Write(size_t offset, const BindingDesc &binding, const std::vector<BufferView> &values);
    bool Write(size_t offset, const BindingDesc &binding, const std::vector<IImage *> &values);
    bool Write(size_t offset, const BindingDesc &binding, const std::vector<SamplerDesc> &values);

   protected:
    size_t SlotCount() const;
    uint32_t ArrayElement(size_t slot) const;
    virtual ObjectPtr<Version> CreateEmpty() const = 0;
    virtual bool WriteDescriptors(size_t offset, const BindingDesc &binding, const std::vector<BufferView> &values) = 0;
    virtual bool WriteDescriptors(size_t offset, const BindingDesc &binding, const std::vector<IImage *> &values) = 0;
    virtual bool WriteDescriptors(size_t offset,
                                  const BindingDesc &binding,
                                  const std::vector<SamplerDesc> &values) = 0;
    Device *device_;
    BindingLayoutDesc layout_;

   private:
    // Slots map into compact, type-specific arrays; layout fixes each slot's type.
    std::vector<size_t> typed_offsets_;
    std::vector<BufferView> buffers_;
    std::vector<IImage *> images_;
    std::vector<SamplerDesc> samplers_;
    std::vector<bool> initialized_;
    std::vector<ObjectPtr<IObject>> references_;
  };

  BindingGroup(Device *device, ObjectPtr<Version> version);
  using IBindingGroup::BindResource;
  const BindingLayoutDesc &GetLayout() const override;
  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) override;
  bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) override;
  bool BindResource(uint32_t binding, const std::vector<SamplerDesc> &samplers, uint32_t first_element = 0) override;
  Device *Owner() const;
  // Borrowed until the next mutation; the context retains it before recording use.
  Version *CurrentVersion() const;

 private:
  const BindingDesc *FindBinding(uint32_t binding, uint32_t first_element, size_t count, size_t &offset) const;
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
