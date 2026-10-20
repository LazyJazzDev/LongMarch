#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"

#include <algorithm>

#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/rhi_buffer.h"

namespace grassland::rhi::backend::d3d12 {
D3D12BindingGroup::Version::Version(D3D12Device *device, const BindingLayoutDesc &layout)
    : BindingGroup::Version(device, layout) {
}

ObjectPtr<BindingGroup::Version> D3D12BindingGroup::Version::CreateEmpty() const {
  return ObjectPtr<Version>::Adopt(new Version(static_cast<D3D12Device *>(device_), layout_));
}

bool D3D12BindingGroup::Version::WriteDescriptors(size_t, const std::vector<Resource> &resources) {
  for (auto e : resources) {
    if (e.image && !dynamic_cast<D3D12AllocatedImage *>(e.image))
      return false;
    auto b = *std::find_if(layout_.bindings.begin(), layout_.bindings.end(),
                           [e](auto binding) { return binding.binding == e.binding; });
    if (b.type == BINDING_UNIFORM_BUFFER) {
      if (e.size > 65536)
        return false;
      const auto size = (e.size + 255) & ~uint64_t(255);
      if (e.offset % 256 || size > e.buffer->GetDesc().size - e.offset)
        return false;
    }
  }
  // The context packs captured versions into its shader-visible descriptor tables.
  return true;
}

D3D12BindingGroup::D3D12BindingGroup(D3D12Device *device, ObjectPtr<Version> version)
    : BindingGroup(device, std::move(version)) {
}

D3D12BindingGroup *D3D12BindingGroup::Create(D3D12Device *device, const BindingLayoutDesc &layout) {
  try {
    if (!ValidateBindingLayout(layout))
      return nullptr;
    auto version = ObjectPtr<Version>::Adopt(new Version(device, layout));
    return new D3D12BindingGroup(device, std::move(version));
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
}  // namespace grassland::rhi::backend::d3d12
