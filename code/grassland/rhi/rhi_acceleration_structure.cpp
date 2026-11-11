#include "grassland/rhi/rhi_acceleration_structure.h"

namespace grassland::rhi {
IAccelerationStructure::~IAccelerationStructure() = default;
IBottomLevelAccelerationStructure::~IBottomLevelAccelerationStructure() = default;
ITopLevelAccelerationStructure::~ITopLevelAccelerationStructure() = default;
}  // namespace grassland::rhi

LONGMARCH_RHI_UUID(grassland::rhi::IAccelerationStructure, "538da216-af18-4ea2-a045-a5350c2000ed")

LONGMARCH_RHI_UUID(grassland::rhi::IBottomLevelAccelerationStructure, "bef07d23-d6c1-47c3-8dab-8557ca3d12ba")

LONGMARCH_RHI_UUID(grassland::rhi::ITopLevelAccelerationStructure, "924ef674-57f1-493d-8d53-561acc912034")
