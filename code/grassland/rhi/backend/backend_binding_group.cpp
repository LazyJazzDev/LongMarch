#include "grassland/rhi/backend/backend_binding_group.h"

#include <algorithm>
#include <set>

#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/rhi_acceleration_structure.h"

namespace grassland::rhi::backend {
std::vector<MappedBinding> MapBindings(const PipelineLayoutDesc &layout, BackendAPI api) {
  std::vector<MappedBinding> result;
  for (uint32_t group = 0; group < layout.groups.size(); ++group) {
    auto bindings = layout.groups[group].bindings;
    std::sort(bindings.begin(), bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
    uint32_t registers[4]{}, resource_offset = 0, sampler_offset = 0;
    for (auto b : bindings) {
      uint32_t kind = b.type == BINDING_UNIFORM_BUFFER ? 0
                      : b.type == BINDING_SAMPLER      ? 3
                      : (b.type == BINDING_RW_STRUCTURED_BUFFER || b.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
                         b.type == BINDING_RW_TEXTURE2D)
                          ? 2
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
    if (b.type < BINDING_UNIFORM_BUFFER || b.type > BINDING_ACCELERATION_STRUCTURE || !b.count || !b.visibility ||
        (b.visibility & ~SHADER_VISIBILITY_ALL) || !bindings.insert(b.binding).second)
      return false;
    if ((b.type == BINDING_STRUCTURED_BUFFER || b.type == BINDING_RW_STRUCTURED_BUFFER) && !b.stride)
      return false;
  }
  return true;
}

bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b) {
  if (a.bindings.size() != b.bindings.size())
    return false;
  for (auto x : a.bindings) {
    auto it = std::find_if(b.bindings.begin(), b.bindings.end(), [x](auto y) { return x.binding == y.binding; });
    if (it == b.bindings.end() || x.type != it->type || x.count != it->count || x.visibility != it->visibility ||
        ((x.type == BINDING_STRUCTURED_BUFFER || x.type == BINDING_RW_STRUCTURED_BUFFER) && x.stride != it->stride))
      return false;
  }
  return true;
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendBindingVersion, "8b356457-129b-40a7-ab14-51c6b2565805")

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendBindingGroup, "84d77c02-7724-4c23-ab73-333b6878b5aa")
