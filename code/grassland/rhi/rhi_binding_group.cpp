#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi {
IBindingGroup::~IBindingGroup() = default;

BindingGroup::BindingGroup(ObjPtr<IBindingGroup> group) noexcept : ObjPtr<IBindingGroup>(std::move(group)) {
}

BindingGroup &BindingGroup::operator=(BindingGroup group) noexcept {
  ObjPtr<IBindingGroup>::operator=(std::move(group));
  return *this;
}

BindingGroup BindingGroup::Retain(IBindingGroup *group) noexcept {
  return ObjPtr<IBindingGroup>::Retain(group);
}

bool BindingGroup::BindResource(uint32_t binding, IBuffer *buffer) const {
  return BindResource(binding, BufferView{buffer});
}

bool BindingGroup::BindResource(uint32_t binding, const BufferView &buffer) const {
  return BindResource(binding, std::vector<BufferView>{buffer});
}

bool BindingGroup::BindResource(uint32_t binding, IImage *image) const {
  return BindResource(binding, ImageView{image});
}

bool BindingGroup::BindResource(uint32_t binding, const ImageView &image) const {
  return BindResource(binding, std::vector<ImageView>{image});
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element) const {
  std::vector<ImageView> views;
  views.reserve(images.size());
  for (auto image : images)
    views.push_back({image});
  return BindResource(binding, views, first_element);
}

bool BindingGroup::BindResource(uint32_t binding, ISampler *sampler) const {
  return BindResource(binding, std::vector<ISampler *>{sampler});
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<IBuffer *> &buffers, uint32_t first_element) const {
  std::vector<BufferView> views;
  views.reserve(buffers.size());
  for (auto buffer : buffers)
    views.push_back({buffer});
  return BindResource(binding, views, first_element);
}

bool BindingGroup::BindResource(uint32_t binding, ITopLevelAccelerationStructure *structure) const {
  return BindResource(binding, std::vector<ITopLevelAccelerationStructure *>{structure});
}

bool BindingGroup::BindResource(uint32_t binding,
                                const std::vector<BufferView> &buffers,
                                uint32_t first_element) const {
  return Get()->BindResource(binding, buffers, first_element);
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<ImageView> &images, uint32_t first_element) const {
  return Get()->BindResource(binding, images, first_element);
}

bool BindingGroup::BindResource(uint32_t binding,
                                const std::vector<ISampler *> &samplers,
                                uint32_t first_element) const {
  return Get()->BindResource(binding, samplers, first_element);
}

bool BindingGroup::BindResource(uint32_t binding,
                                const std::vector<ITopLevelAccelerationStructure *> &structures,
                                uint32_t first_element) const {
  return Get()->BindResource(binding, structures, first_element);
}

}  // namespace grassland::rhi
