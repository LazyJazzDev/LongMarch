#include "sparkium/pipelines/raytracing/core/native_tlas.h"

#include <algorithm>
#include <limits>

namespace sparkium::raytracing {
using namespace grassland::rhi;

bool NativeTLAS::Record(IDevice *device,
                        ICommandContext *commands,
                        ResourceUpdates::Cursor &updates,
                        const std::vector<RayTracingInstance> &instances) {
  if (!device || !commands || instances.size() > std::numeric_limits<uint32_t>::max())
    return false;
  if (!structure_ || structure_->GetDesc().instance_count < instances.size())
    structure_ = device->CreateTopLevelAccelerationStructure(
        {static_cast<uint32_t>(std::max<size_t>(1, instances.size())), true});
  if (!structure_)
    return false;
  const bool compatible =
      accepted_->structure.Get() == structure_.Get() && accepted_->instances.size() == instances.size();
  const auto same = [&](const RayTracingInstance &a, const RayTracingInstance &b) {
    return a.blas == b.blas && !updates.Builds(a.blas) && a.transform == b.transform &&
           a.custom_index == b.custom_index && a.hit_group_offset == b.hit_group_offset && a.mask == b.mask &&
           a.disable_triangle_culling == b.disable_triangle_culling &&
           a.front_counter_clockwise == b.front_counter_clockwise;
  };
  if (compatible && std::equal(instances.begin(), instances.end(), accepted_->instances.begin(), same))
    return true;
  if (!commands->CmdBuildTopLevelAccelerationStructure(
          structure_.Get(), instances, compatible ? ACCELERATION_STRUCTURE_UPDATE : ACCELERATION_STRUCTURE_BUILD))
    return false;
  State recorded{structure_, instances};
  updates.OnAccepted([state = accepted_, recorded = std::move(recorded)]() mutable { *state = std::move(recorded); });
  return true;
}

ITopLevelAccelerationStructure *NativeTLAS::Get() const {
  return structure_.Get();
}

}  // namespace sparkium::raytracing
