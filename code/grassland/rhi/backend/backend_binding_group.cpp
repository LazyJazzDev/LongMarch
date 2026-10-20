#include "grassland/rhi/backend/backend_binding_group.h"

#include <algorithm>
#include <set>

#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {
std::vector<MappedBinding> MapBindings(const PipelineLayoutDesc &layout, BackendAPI api) {
  std::vector<MappedBinding> result;
  for (uint32_t group = 0; group < layout.groups.size(); ++group) {
    auto bindings = layout.groups[group].bindings;
    std::sort(bindings.begin(), bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
    uint32_t registers[4]{}, resource_offset = 0, sampler_offset = 0;
    for (auto b : bindings) {
      uint32_t kind = b.type == BINDING_UNIFORM_BUFFER                                          ? 0
                      : b.type == BINDING_SAMPLER                                               ? 3
                      : (b.type == BINDING_RW_STORAGE_BUFFER || b.type == BINDING_RW_TEXTURE2D) ? 2
                                                                                                : 1;
      uint32_t reg = registers[kind];
      registers[kind] += b.count;
      if (api == BACKEND_API_VULKAN)
        reg = b.binding;
      auto &offset = b.type == BINDING_SAMPLER ? sampler_offset : resource_offset;
      result.push_back({b, group, reg, offset});
      offset += b.count;
    }
  }
  return result;
}

bool ValidateBindingLayout(const BindingLayoutDesc &layout) {
  std::set<uint32_t> bindings;
  uint64_t total = 0;
  for (auto b : layout.bindings) {
    if (b.type < BINDING_UNIFORM_BUFFER || b.type > BINDING_SAMPLER || !b.count || !b.visibility ||
        (b.visibility & ~SHADER_VISIBILITY_ALL) || !bindings.insert(b.binding).second)
      return false;
    total += b.count;
  }
  return total <= 1024;
}

bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b) {
  if (a.bindings.size() != b.bindings.size())
    return false;
  for (auto x : a.bindings) {
    auto it = std::find_if(b.bindings.begin(), b.bindings.end(), [x](auto y) { return x.binding == y.binding; });
    if (it == b.bindings.end() || x.type != it->type || x.count != it->count || x.visibility != it->visibility)
      return false;
  }
  return true;
}

bool BindingGroup::Validate(Device *device, const BindingLayoutDesc &layout, const std::vector<BindingEntry> &entries) {
  if (!ValidateBindingLayout(layout))
    return false;
  size_t expected = 0;
  for (auto b : layout.bindings)
    expected += b.count;
  if (expected != entries.size())
    return false;
  std::set<std::pair<uint32_t, uint32_t>> indices;
  for (auto e : entries) {
    auto it =
        std::find_if(layout.bindings.begin(), layout.bindings.end(), [e](auto b) { return b.binding == e.binding; });
    if (it == layout.bindings.end() || e.array_element >= it->count ||
        !indices.emplace(e.binding, e.array_element).second)
      return false;
    if (it->type <= BINDING_RW_STORAGE_BUFFER) {
      auto buffer = dynamic_cast<Buffer *>(e.buffer);
      if (e.image || !buffer || buffer->Owner() != device || !e.size || e.offset > buffer->GetDesc().size ||
          e.size > buffer->GetDesc().size - e.offset)
        return false;
      if (it->type != BINDING_UNIFORM_BUFFER && (!e.stride || e.offset % e.stride || e.size % e.stride || e.stride % 4))
        return false;
    } else if (it->type == BINDING_TEXTURE2D || it->type == BINDING_RW_TEXTURE2D) {
      auto image = dynamic_cast<Image *>(e.image);
      if (e.buffer || !image || !image->IsValid() || image->DeviceIdentity() != device->NativeIdentity() ||
          IsDepthFormat(image->GetDesc().format))
        return false;
      if (it->type == BINDING_RW_TEXTURE2D && !image->SupportsStorage())
        return false;
    } else if (e.buffer || e.image || e.sampler.filter < SAMPLER_FILTER_NEAREST ||
               e.sampler.filter > SAMPLER_FILTER_LINEAR || e.sampler.address_u < SAMPLER_ADDRESS_REPEAT ||
               e.sampler.address_u > SAMPLER_ADDRESS_MIRRORED_REPEAT || e.sampler.address_v < SAMPLER_ADDRESS_REPEAT ||
               e.sampler.address_v > SAMPLER_ADDRESS_MIRRORED_REPEAT)
      return false;
  }
  return true;
}

BindingGroup::BindingGroup(Device *device, const BindingLayoutDesc &layout, const std::vector<BindingEntry> &entries)
    : device_(device),
      layout_(layout),
      entries_(entries) {
  std::sort(layout_.bindings.begin(), layout_.bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
  std::sort(entries_.begin(), entries_.end(), [](auto a, auto b) {
    return std::make_pair(a.binding, a.array_element) < std::make_pair(b.binding, b.array_element);
  });
  for (auto e : entries_) {
    if (e.buffer)
      resources_.push_back(ObjectPtr<IObject>::Retain(e.buffer));
    if (e.image)
      resources_.push_back(ObjectPtr<IObject>::Retain(e.image));
  }
}

const BindingLayoutDesc &BindingGroup::GetLayout() const {
  return layout_;
}

const std::vector<BindingEntry> &BindingGroup::GetEntries() const {
  return entries_;
}

Device *BindingGroup::Owner() const {
  return device_;
}
}  // namespace grassland::rhi::backend
