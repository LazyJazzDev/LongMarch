#include "grassland/rhi/backend/backend_acceleration_structure.h"

#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend {
bool ValidateBottomLevelDesc(const BottomLevelAccelerationStructureDesc &desc) {
  if (desc.geometries.empty() == desc.aabbs.empty() || desc.geometries.size() > UINT32_MAX ||
      desc.aabbs.size() > UINT32_MAX)
    return false;
  for (const auto &g : desc.geometries) {
    if (!g.vertex_count || g.vertex_stride < 12 || g.vertex_stride % 4)
      return false;
    if ((g.index_count ? g.index_count : g.vertex_count) % 3)
      return false;
    if (g.index_count && g.index_format != INDEX_FORMAT_UINT16 && g.index_format != INDEX_FORMAT_UINT32)
      return false;
  }
  for (const auto &g : desc.aabbs)
    if (!g.count || g.stride < 24 || g.stride % 8)
      return false;
  return true;
}

bool ValidateGeometryBuffers(const BottomLevelAccelerationStructureDesc &desc,
                             const std::vector<AccelerationStructureGeometryBuffers> &buffers,
                             Device *device) {
  if (buffers.size() != desc.geometries.size() + desc.aabbs.size())
    return false;
  auto valid = [device](const BufferView &view, uint64_t required, uint64_t alignment) {
    auto buffer = dynamic_cast<Buffer *>(view.buffer);
    if (!buffer || buffer->Owner() != device || view.offset % alignment)
      return false;
    auto size = buffer->GetDesc().size;
    if (view.offset > size || required > size - view.offset)
      return false;
    return !view.size || (view.size <= size - view.offset && required <= view.size);
  };
  for (size_t i = 0; i < desc.geometries.size(); ++i) {
    const auto &g = desc.geometries[i];
    if (!valid(buffers[i].data, uint64_t(g.vertex_count - 1) * g.vertex_stride + 12, 4))
      return false;
    uint64_t index_size = g.index_format == INDEX_FORMAT_UINT16 ? 2 : 4;
    if (g.index_count && !valid(buffers[i].indices, uint64_t(g.index_count) * index_size, index_size))
      return false;
  }
  for (size_t i = 0; i < desc.aabbs.size(); ++i)
    if (!valid(buffers[i].data, uint64_t(desc.aabbs[i].count - 1) * desc.aabbs[i].stride + 24, 8))
      return false;
  return true;
}

bool ValidateInstances(const TopLevelAccelerationStructureDesc &desc,
                       const std::vector<RayTracingInstance> &instances) {
  if (instances.size() != desc.instance_count)
    return false;
  for (const auto &instance : instances)
    if (!instance.blas || instance.custom_index > 0xffffff || instance.hit_group_offset > 0xffffff)
      return false;
  return true;
}
}  // namespace grassland::rhi::backend
