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
  for (auto b : layout.bindings) {
    if (b.type < BINDING_UNIFORM_BUFFER || b.type > BINDING_SAMPLER || !b.count || !b.visibility ||
        (b.visibility & ~SHADER_VISIBILITY_ALL) || !bindings.insert(b.binding).second)
      return false;
  }
  return true;
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

BindingGroup::Version::Version(Device *device, const BindingLayoutDesc &layout) : device_(device), layout_(layout) {
  std::sort(layout_.bindings.begin(), layout_.bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
  for (auto b : layout_.bindings)
    for (uint32_t i = 0; i < b.count; ++i) {
      Resource resource;
      resource.binding = b.binding;
      resource.array_element = i;
      resources_.push_back(resource);
    }
  references_.resize(resources_.size());
}

const BindingLayoutDesc &BindingGroup::Version::GetLayout() const {
  return layout_;
}

const std::vector<BindingGroup::Resource> &BindingGroup::Version::GetResources() const {
  return resources_;
}

bool BindingGroup::Version::IsComplete() const {
  return std::all_of(resources_.begin(), resources_.end(), [](const auto &r) { return r.initialized; });
}

ObjectPtr<BindingGroup::Version> BindingGroup::Version::Clone() const {
  auto copy = CreateEmpty();
  if (!copy)
    return {};
  if (IsComplete()) {
    if (!resources_.empty())
      if (!copy->Write(0, resources_))
        return {};
    return copy;
  }
  // Only initialized slots may be copied into a native descriptor set.
  for (size_t i = 0; i < resources_.size(); ++i)
    if (resources_[i].initialized)
      if (!copy->Write(i, {resources_[i]}))
        return {};
  return copy;
}

bool BindingGroup::Version::Write(size_t offset, const std::vector<Resource> &resources) {
  // Retain the whole update before replacing any slot: resources may be swapped
  // between slots while the group is their only owner.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(resources.size());
  for (const auto &r : resources) {
    IObject *object = r.buffer ? static_cast<IObject *>(r.buffer) : static_cast<IObject *>(r.image);
    references.push_back(ObjectPtr<IObject>::Retain(object));
  }
  if (!WriteDescriptors(offset, resources))
    return false;
  for (size_t i = 0; i < resources.size(); ++i) {
    references_[offset + i] = std::move(references[i]);
    resources_[offset + i] = resources[i];
  }
  return true;
}

BindingGroup::BindingGroup(Device *device, ObjectPtr<Version> version)
    : device_(device),
      layout_(version->GetLayout()),
      version_(std::move(version)) {
}

const BindingLayoutDesc &BindingGroup::GetLayout() const {
  return layout_;
}

Device *BindingGroup::Owner() const {
  return device_;
}

BindingGroup::Version *BindingGroup::CurrentVersion() const {
  return version_.Get();
}

bool BindingGroup::Write(uint32_t binding, uint32_t first_element, BindingType kind, std::vector<Resource> resources) {
  const auto &layout = GetLayout();
  size_t offset = 0;
  auto it = layout.bindings.begin();
  for (; it != layout.bindings.end(); ++it) {
    if (it->binding == binding)
      break;
    offset += it->count;
  }
  if (it == layout.bindings.end() || resources.empty() || first_element > it->count ||
      resources.size() > it->count - first_element)
    return false;
  if (kind == BINDING_UNIFORM_BUFFER) {
    if (it->type > BINDING_RW_STORAGE_BUFFER)
      return false;
  } else if (kind == BINDING_TEXTURE2D) {
    if (it->type != BINDING_TEXTURE2D && it->type != BINDING_RW_TEXTURE2D)
      return false;
  } else if (it->type != BINDING_SAMPLER)
    return false;
  for (size_t i = 0; i < resources.size(); ++i) {
    auto &e = resources[i];
    e.binding = binding;
    e.array_element = first_element + uint32_t(i);
    e.initialized = true;
    if (kind == BINDING_UNIFORM_BUFFER) {
      auto buffer = dynamic_cast<Buffer *>(e.buffer);
      if (!buffer || buffer->Owner() != device_ || e.offset >= buffer->GetDesc().size)
        return false;
      if (!e.size)
        e.size = buffer->GetDesc().size - e.offset;
      if (e.size > buffer->GetDesc().size - e.offset)
        return false;
      if (it->type != BINDING_UNIFORM_BUFFER && (!e.stride || e.offset % e.stride || e.size % e.stride || e.stride % 4))
        return false;
    } else if (kind == BINDING_TEXTURE2D) {
      auto image = dynamic_cast<Image *>(e.image);
      if (!image || !image->IsValid() || image->DeviceIdentity() != device_->NativeIdentity() ||
          IsDepthFormat(image->GetDesc().format))
        return false;
      if (it->type == BINDING_RW_TEXTURE2D && !image->SupportsStorage())
        return false;
    } else if (e.sampler.filter < SAMPLER_FILTER_NEAREST || e.sampler.filter > SAMPLER_FILTER_LINEAR ||
               e.sampler.address_u < SAMPLER_ADDRESS_REPEAT || e.sampler.address_u > SAMPLER_ADDRESS_MIRRORED_REPEAT ||
               e.sampler.address_v < SAMPLER_ADDRESS_REPEAT || e.sampler.address_v > SAMPLER_ADDRESS_MIRRORED_REPEAT)
      return false;
  }
  offset += first_element;
  if (version_->IsUniquelyOwned())
    return version_->Write(offset, resources);
  auto next = version_->Clone();
  if (!next)
    return false;
  if (!next->Write(offset, resources))
    return false;
  version_ = std::move(next);
  return true;
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element) {
  try {
    std::vector<Resource> resources;
    resources.reserve(buffers.size());
    for (auto b : buffers) {
      Resource r;
      r.buffer = b.buffer;
      r.offset = b.offset;
      r.size = b.size;
      r.stride = b.stride;
      resources.push_back(r);
    }
    return Write(binding, first_element, BINDING_UNIFORM_BUFFER, std::move(resources));
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element) {
  try {
    std::vector<Resource> resources;
    resources.reserve(images.size());
    for (auto image : images) {
      Resource r;
      r.image = image;
      resources.push_back(r);
    }
    return Write(binding, first_element, BINDING_TEXTURE2D, std::move(resources));
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<SamplerDesc> &samplers, uint32_t first_element) {
  try {
    std::vector<Resource> resources;
    resources.reserve(samplers.size());
    for (auto sampler : samplers) {
      Resource r;
      r.sampler = sampler;
      resources.push_back(r);
    }
    return Write(binding, first_element, BINDING_SAMPLER, std::move(resources));
  } catch (const std::bad_alloc &) {
    return false;
  }
}
}  // namespace grassland::rhi::backend
