#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend {
class Device;

class BindingGroup : public Object<IBindingGroup> {
 public:
  // Backend-only slot encoding, never supplied by callers.
  struct Resource {
    uint32_t binding{}, array_element{};
    IBuffer *buffer{};
    uint64_t offset{}, size{};
    IImage *image{};
    SamplerDesc sampler;
    bool initialized{};
  };

  class Version : public Object<IObject> {
   public:
    Version(Device *device, const BindingLayoutDesc &layout);
    const BindingLayoutDesc &GetLayout() const;
    const std::vector<Resource> &GetResources() const;
    bool IsComplete() const;
    ObjectPtr<Version> Clone() const;
    bool Write(size_t offset, const std::vector<Resource> &resources);

   protected:
    virtual ObjectPtr<Version> CreateEmpty() const = 0;
    virtual bool WriteDescriptors(size_t offset, const std::vector<Resource> &resources) = 0;
    Device *device_;
    BindingLayoutDesc layout_;

   private:
    std::vector<Resource> resources_;
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
  bool Write(uint32_t binding, uint32_t first_element, BindingType kind, std::vector<Resource> resources);
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
