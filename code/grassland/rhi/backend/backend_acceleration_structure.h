#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"

namespace grassland::rhi::backend {

bool ValidateBottomLevelDesc(const BottomLevelAccelerationStructureDesc &desc);
bool ValidateGeometryBuffers(const BottomLevelAccelerationStructureDesc &desc,
                             const std::vector<AccelerationStructureGeometryBuffers> &buffers,
                             IBackendDevice *device);
bool ValidateInstances(const TopLevelAccelerationStructureDesc &desc, const std::vector<RayTracingInstance> &instances);
}  // namespace grassland::rhi::backend
