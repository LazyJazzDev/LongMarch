#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi {
bool IBindingGroup::BindResource(uint32_t binding, IBuffer *buffer) {
  return BindResource(binding, BufferView{buffer});
}

bool IBindingGroup::BindResource(uint32_t binding, const BufferView &buffer) {
  try {
    return BindResource(binding, std::vector<BufferView>{buffer});
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool IBindingGroup::BindResource(uint32_t binding, IImage *image) {
  try {
    return BindResource(binding, std::vector<IImage *>{image});
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool IBindingGroup::BindResource(uint32_t binding, const SamplerDesc &sampler) {
  try {
    return BindResource(binding, std::vector<SamplerDesc>{sampler});
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool IBindingGroup::BindResource(uint32_t binding, const std::vector<IBuffer *> &buffers, uint32_t first_element) {
  try {
    std::vector<BufferView> views;
    views.reserve(buffers.size());
    for (auto buffer : buffers)
      views.push_back({buffer});
    return BindResource(binding, views, first_element);
  } catch (const std::bad_alloc &) {
    return false;
  }
}
}  // namespace grassland::rhi
