#include "grassland/rhi/rhix/rhix_binding_group.h"

namespace grassland::rhix {
BindingGroup::BindingGroup() noexcept = default;

BindingGroup::BindingGroup(rhi::IBindingGroup *object) noexcept : ObjectContainer(object) {
}

rhi::IBindingGroup *BindingGroup::Get() const noexcept {
  return static_cast<rhi::IBindingGroup *>(GetObject());
}

const rhi::BindingLayoutDesc &BindingGroup::GetLayout() const {
  return Get()->GetLayout();
}

bool BindingGroup::BindResource(uint32_t binding, const Buffer &buffer) {
  return Get()->BindResource(binding, buffer.Get());
}

bool BindingGroup::BindResource(uint32_t binding, const BufferView &buffer) {
  return Get()->BindResource(binding, rhi::BufferView{buffer.buffer.Get(), buffer.offset, buffer.size, buffer.stride});
}

bool BindingGroup::BindResource(uint32_t binding, const Image &image) {
  return Get()->BindResource(binding, image.Get());
}

bool BindingGroup::BindResource(uint32_t binding, const rhi::SamplerDesc &sampler) {
  return Get()->BindResource(binding, sampler);
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<Buffer> &buffers, uint32_t first_element) {
  try {
    std::vector<rhi::IBuffer *> native;
    native.reserve(buffers.size());
    for (const auto &buffer : buffers)
      native.push_back(buffer.Get());
    return Get()->BindResource(binding, native, first_element);
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element) {
  try {
    std::vector<rhi::BufferView> native;
    native.reserve(buffers.size());
    for (const auto &b : buffers)
      native.push_back({b.buffer.Get(), b.offset, b.size, b.stride});
    return Get()->BindResource(binding, native, first_element);
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<Image> &images, uint32_t first_element) {
  try {
    std::vector<rhi::IImage *> native;
    native.reserve(images.size());
    for (const auto &image : images)
      native.push_back(image.Get());
    return Get()->BindResource(binding, native, first_element);
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding,
                                const std::vector<rhi::SamplerDesc> &samplers,
                                uint32_t first_element) {
  return Get()->BindResource(binding, samplers, first_element);
}
}  // namespace grassland::rhix
