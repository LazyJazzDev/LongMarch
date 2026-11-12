#pragma once
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalBindingVersionImpl final : public Object<IMetalBindingVersion> {
 public:
  static MetalBindingVersionImpl *Create(IMetalDevice *device, const BindingLayoutDesc &layout);
  ObjectPtr<IMetalBindingVersion> Clone() const override;
  bool IsInitialized(size_t binding, uint32_t element) const override;
  void Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const;
  void Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const;

 private:
  IMetalDevice *device_;
  MetalBindingVersionImpl(IMetalDevice *device, const BindingLayoutDesc &layout);
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<BufferView> &values);
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<IImage *> &values);
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<ISampler *> &values);
  bool WriteDescriptors(size_t binding, uint32_t first, const std::vector<ITopLevelAccelerationStructure *> &values);
  std::vector<uint32_t> offsets_;
  std::vector<std::vector<bool>> initialized_;
  NS::SharedPtr<MTL::ArgumentEncoder> encoder_;
  NS::SharedPtr<MTL::Buffer> arguments_, sizes_;

 public:
  ~MetalBindingVersionImpl() override = default;

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

class MetalBindingGroupImpl final : public Object<rhi::IBindingGroup, IMetalBindingGroup> {
 public:
  using Version = IMetalBindingVersion;
  static IBindingGroup *Create(IMetalDevice *device, const BindingLayoutDesc &layout);

 private:
  MetalBindingGroupImpl(IMetalDevice *device, ObjectPtr<Version> version);

 public:
  ~MetalBindingGroupImpl() override = default;

 private:
  void InitializeBindingGroup(IMetalDevice *device, ObjectPtr<IMetalBindingVersion> version);

 public:
  const BindingLayoutDesc &GetLayout() const override;

  IMetalDevice *Owner() const override;

  // Borrowed until the next mutation; the context retains it before recording use.
  IMetalBindingVersion *CurrentVersion() const override;

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
  IMetalDevice *device_;
  BindingLayoutDesc layout_;
  ObjectPtr<IMetalBindingVersion> version_;
};
}  // namespace grassland::rhi::backend::metal
