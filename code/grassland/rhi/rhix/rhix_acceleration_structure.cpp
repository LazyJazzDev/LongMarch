#include "grassland/rhi/rhix/rhix_acceleration_structure.h"

namespace grassland::rhix {
AccelerationStructure::AccelerationStructure() noexcept = default;

AccelerationStructure::AccelerationStructure(rhi::IAccelerationStructure *object) noexcept : ObjectContainer(object) {
}

rhi::IAccelerationStructure *AccelerationStructure::Get() const noexcept {
  return static_cast<rhi::IAccelerationStructure *>(GetObject());
}

rhi::AccelerationStructureType AccelerationStructure::GetType() const {
  return Get()->GetType();
}

uint64_t AccelerationStructure::GetCompactedSize() const {
  return Get()->GetCompactedSize();
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure() noexcept = default;

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(
    rhi::IBottomLevelAccelerationStructure *object) noexcept
    : AccelerationStructure(object) {
}

rhi::IBottomLevelAccelerationStructure *BottomLevelAccelerationStructure::Get() const noexcept {
  return static_cast<rhi::IBottomLevelAccelerationStructure *>(GetObject());
}

const rhi::BottomLevelAccelerationStructureDesc &BottomLevelAccelerationStructure::GetDesc() const {
  return Get()->GetDesc();
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure() noexcept = default;

TopLevelAccelerationStructure::TopLevelAccelerationStructure(rhi::ITopLevelAccelerationStructure *object) noexcept
    : AccelerationStructure(object) {
}

rhi::ITopLevelAccelerationStructure *TopLevelAccelerationStructure::Get() const noexcept {
  return static_cast<rhi::ITopLevelAccelerationStructure *>(GetObject());
}

const rhi::TopLevelAccelerationStructureDesc &TopLevelAccelerationStructure::GetDesc() const {
  return Get()->GetDesc();
}
}  // namespace grassland::rhix
