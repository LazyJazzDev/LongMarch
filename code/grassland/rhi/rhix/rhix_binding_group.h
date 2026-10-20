#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhix/rhix_buffer.h"
#include "grassland/rhi/rhix/rhix_image.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
struct BufferView {
  Buffer buffer;
  uint64_t offset{}, size{};
  uint32_t stride{4};
};

class BindingGroup : public ObjectContainer {
 public:
  BindingGroup() noexcept;
  BindingGroup(rhi::IBindingGroup *object) noexcept;
  rhi::IBindingGroup *Get() const noexcept;
  const rhi::BindingLayoutDesc &GetLayout() const;
  bool BindResource(uint32_t binding, const Buffer &buffer);
  bool BindResource(uint32_t binding, const BufferView &buffer);
  bool BindResource(uint32_t binding, const Image &image);
  bool BindResource(uint32_t binding, const rhi::SamplerDesc &sampler);
  bool BindResource(uint32_t binding, const std::vector<Buffer> &buffers, uint32_t first_element = 0);
  bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0);
  bool BindResource(uint32_t binding, const std::vector<Image> &images, uint32_t first_element = 0);
  bool BindResource(uint32_t binding, const std::vector<rhi::SamplerDesc> &samplers, uint32_t first_element = 0);
};
}  // namespace grassland::rhix
