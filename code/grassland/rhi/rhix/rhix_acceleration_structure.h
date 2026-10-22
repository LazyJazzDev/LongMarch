#pragma once
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhix/rhix_buffer.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class AccelerationStructure : public ObjectContainer {
 public:
  AccelerationStructure() noexcept;
  AccelerationStructure(rhi::IAccelerationStructure *object) noexcept;
  rhi::IAccelerationStructure *Get() const noexcept;
  rhi::AccelerationStructureType GetType() const;
  uint64_t GetCompactedSize() const;
};

class BottomLevelAccelerationStructure : public AccelerationStructure {
 public:
  BottomLevelAccelerationStructure() noexcept;
  BottomLevelAccelerationStructure(rhi::IBottomLevelAccelerationStructure *object) noexcept;
  rhi::IBottomLevelAccelerationStructure *Get() const noexcept;
  const rhi::BottomLevelAccelerationStructureDesc &GetDesc() const;
};

class TopLevelAccelerationStructure : public AccelerationStructure {
 public:
  TopLevelAccelerationStructure() noexcept;
  TopLevelAccelerationStructure(rhi::ITopLevelAccelerationStructure *object) noexcept;
  rhi::ITopLevelAccelerationStructure *Get() const noexcept;
  const rhi::TopLevelAccelerationStructureDesc &GetDesc() const;
};

struct AccelerationStructureGeometryBuffers {
  BufferView data;
  BufferView indices;
};

struct RayTracingInstance {
  BottomLevelAccelerationStructure blas;
  std::array<float, 12> transform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  uint32_t custom_index{}, hit_group_offset{};
  uint8_t mask{0xff};
  bool disable_triangle_culling{}, front_counter_clockwise{};
};
}  // namespace grassland::rhix
