#include "grassland/rhi/rhi_acceleration_structure.h"

namespace grassland::rhi {
IAccelerationStructure::~IAccelerationStructure() = default;
IBottomLevelAccelerationStructure::~IBottomLevelAccelerationStructure() = default;
ITopLevelAccelerationStructure::~ITopLevelAccelerationStructure() = default;
}  // namespace grassland::rhi
