#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend {
class Device;

class BindingGroup : public Object<IBindingGroup> {
 public:
  LONGMARCH_RHI_QUERY(BindingGroup, Object<IBindingGroup>)

  class Version : public Object<IObject> {
   public:
    using Object<IObject>::QueryInterface;
    bool QueryInterface(InterfaceID id, void **output) noexcept override;

    explicit Version(const BindingLayoutDesc &layout);
    const BindingLayoutDesc &GetLayout() const;
    const std::vector<std::vector<ObjectPtr<IObject>>> &GetReferences() const;
    virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
    bool IsComplete() const;
    virtual ObjectPtr<Version> Clone() const = 0;
    bool Write(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values);
    bool Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values);
    bool Write(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values);
    bool Write(size_t binding_index,
               uint32_t first_element,
               const std::vector<ITopLevelAccelerationStructure *> &values);

   protected:
    virtual bool WriteDescriptors(size_t binding_index,
                                  uint32_t first_element,
                                  const std::vector<ITopLevelAccelerationStructure *> &values);
    virtual bool WriteDescriptors(size_t binding_index,
                                  uint32_t first_element,
                                  const std::vector<BufferView> &values) = 0;
    virtual bool WriteDescriptors(size_t binding_index,
                                  uint32_t first_element,
                                  const std::vector<IImage *> &values) = 0;
    virtual bool WriteDescriptors(size_t binding_index,
                                  uint32_t first_element,
                                  const std::vector<ISampler *> &values) = 0;
    BindingLayoutDesc layout_;
    // Indexed by sorted layout binding, then array element.
    std::vector<std::vector<ObjectPtr<IObject>>> references_;
  };

  BindingGroup(Device *device, ObjectPtr<Version> version);
  using IBindingGroup::BindResource;
  bool BindResource(uint32_t binding,
                    const std::vector<ITopLevelAccelerationStructure *> &structures,
                    uint32_t first_element = 0) override;
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

LONGMARCH_RHI_UUID(grassland::rhi::backend::BindingGroup::Version, "8b356457-129b-40a7-ab14-51c6b2565805")

inline bool grassland::rhi::backend::BindingGroup::Version::QueryInterface(InterfaceID id, void **output) noexcept {
  if (!output)
    return false;
  if (auto result = MatchInterface<Version>(id, this)) {
    *output = result;
    AddRef();
    return true;
  }
  return Object<IObject>::QueryInterface(id, output);
}
