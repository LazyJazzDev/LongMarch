#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi {
bool IBindingGroup::BindResource(uint32_t binding, IBuffer *buffer) {
  return BindResource(binding, BufferView{buffer});
}

bool IBindingGroup::BindResource(uint32_t binding, const BufferView &buffer) {
  return BindResource(binding, std::vector<BufferView>{buffer});
}

bool IBindingGroup::BindResource(uint32_t binding, IImage *image) {
  return BindResource(binding, std::vector<IImage *>{image});
}

bool IBindingGroup::BindResource(uint32_t binding, ISampler *sampler) {
  return BindResource(binding, std::vector<ISampler *>{sampler});
}

bool IBindingGroup::BindResource(uint32_t binding, const std::vector<IBuffer *> &buffers, uint32_t first_element) {
  std::vector<BufferView> views;
  views.reserve(buffers.size());
  for (auto buffer : buffers)
    views.push_back({buffer});
  return BindResource(binding, views, first_element);
}

bool IBindingGroup::BindResource(uint32_t binding, ITopLevelAccelerationStructure *structure) {
  return BindResource(binding, std::vector<ITopLevelAccelerationStructure *>{structure});
}

}  // namespace grassland::rhi
