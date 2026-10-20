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
  size_t offset = 0;
  for (const auto &binding : layout_.bindings) {
    // Copy initialized runs only; unwritten native descriptors must not be read.
    const size_t end = offset + binding.count;
    while (offset < end) {
      if (!resources_[offset].initialized) {
        ++offset;
        continue;
      }
      const size_t first = offset;
      while (offset < end && resources_[offset].initialized)
        ++offset;
      if (binding.type <= BINDING_RW_STORAGE_BUFFER) {
        std::vector<BufferView> buffers;
        for (size_t i = first; i < offset; ++i) {
          const auto &r = resources_[i];
          buffers.push_back({r.buffer, r.offset, r.size});
        }
        if (!copy->Write(first, binding, buffers))
          return {};
      } else if (binding.type == BINDING_SAMPLER) {
        std::vector<SamplerDesc> samplers;
        for (size_t i = first; i < offset; ++i)
          samplers.push_back(resources_[i].sampler);
        if (!copy->Write(first, binding, samplers))
          return {};
      } else {
        std::vector<IImage *> images;
        for (size_t i = first; i < offset; ++i)
          images.push_back(resources_[i].image);
        if (!copy->Write(first, binding, images))
          return {};
      }
    }
  }
  return copy;
}

bool BindingGroup::Version::Write(size_t offset, const BindingDesc &binding, const std::vector<BufferView> &values) {
  // Retain the whole array before replacing slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value.buffer));
  if (!WriteDescriptors(offset, binding, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i) {
    auto &slot = resources_[offset + i];
    slot.buffer = values[i].buffer;
    slot.offset = values[i].offset;
    slot.size = values[i].size;
    slot.initialized = true;
    references_[offset + i] = std::move(references[i]);
  }
  return true;
}

bool BindingGroup::Version::Write(size_t offset, const BindingDesc &binding, const std::vector<IImage *> &values) {
  // Retain the whole array before replacing slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(offset, binding, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i) {
    auto &slot = resources_[offset + i];
    slot.image = values[i];
    slot.initialized = true;
    references_[offset + i] = std::move(references[i]);
  }
  return true;
}

bool BindingGroup::Version::Write(size_t offset, const BindingDesc &binding, const std::vector<SamplerDesc> &values) {
  if (!WriteDescriptors(offset, binding, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i) {
    auto &slot = resources_[offset + i];
    slot.sampler = values[i];
    slot.initialized = true;
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

const BindingDesc *BindingGroup::FindBinding(uint32_t binding,
                                             uint32_t first_element,
                                             size_t count,
                                             size_t &offset) const {
  offset = 0;
  for (const auto &desc : layout_.bindings) {
    if (desc.binding == binding) {
      if (!count || first_element > desc.count || count > desc.count - first_element)
        return nullptr;
      offset += first_element;
      return &desc;
    }
    offset += desc.count;
  }
  return nullptr;
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element) {
  try {
    size_t offset;
    const auto *desc = FindBinding(binding, first_element, buffers.size(), offset);
    if (!desc)
      return false;
    if (desc->type > BINDING_RW_STORAGE_BUFFER)
      return false;
    auto values = buffers;
    for (auto &view : values) {
      auto buffer = dynamic_cast<Buffer *>(view.buffer);
      if (!buffer || buffer->Owner() != device_ || view.offset >= buffer->GetDesc().size)
        return false;
      if (!view.size)
        view.size = buffer->GetDesc().size - view.offset;
      if (view.size > buffer->GetDesc().size - view.offset)
        return false;
    }
    auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
    if (!next)
      return false;
    if (!next->Write(offset, *desc, values))
      return false;
    version_ = std::move(next);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element) {
  try {
    size_t offset;
    const auto *desc = FindBinding(binding, first_element, images.size(), offset);
    if (!desc)
      return false;
    if (desc->type != BINDING_TEXTURE2D && desc->type != BINDING_RW_TEXTURE2D)
      return false;
    for (auto value : images) {
      auto image = dynamic_cast<Image *>(value);
      if (!image || !image->IsValid() || image->DeviceIdentity() != device_->NativeIdentity() ||
          IsDepthFormat(image->GetDesc().format))
        return false;
      if (desc->type == BINDING_RW_TEXTURE2D && !image->SupportsStorage())
        return false;
    }
    auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
    if (!next)
      return false;
    if (!next->Write(offset, *desc, images))
      return false;
    version_ = std::move(next);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool BindingGroup::BindResource(uint32_t binding, const std::vector<SamplerDesc> &samplers, uint32_t first_element) {
  try {
    size_t offset;
    const auto *desc = FindBinding(binding, first_element, samplers.size(), offset);
    if (!desc)
      return false;
    if (desc->type != BINDING_SAMPLER)
      return false;
    for (const auto &sampler : samplers) {
      if (sampler.filter < SAMPLER_FILTER_NEAREST || sampler.filter > SAMPLER_FILTER_LINEAR ||
          sampler.address_u < SAMPLER_ADDRESS_REPEAT || sampler.address_u > SAMPLER_ADDRESS_MIRRORED_REPEAT ||
          sampler.address_v < SAMPLER_ADDRESS_REPEAT || sampler.address_v > SAMPLER_ADDRESS_MIRRORED_REPEAT)
        return false;
    }
    auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
    if (!next)
      return false;
    if (!next->Write(offset, *desc, samplers))
      return false;
    version_ = std::move(next);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

}  // namespace grassland::rhi::backend
