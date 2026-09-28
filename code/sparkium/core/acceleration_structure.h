#pragma once

#include "sparkium/core/data_resource.h"

namespace sparkium {
class BottomLevelAccelerationStructure;

struct AccelerationStructureInstance {
  BottomLevelAccelerationStructure *blas{};
  glm::mat4x3 transform{1.0f};
  uint32_t instance_id{};
  uint32_t instance_mask{255};
  uint32_t instance_hit_group_offset{};
  graphics::RayTracingInstanceFlag instance_flags{graphics::RAYTRACING_INSTANCE_FLAG_NONE};
  bool operator==(const AccelerationStructureInstance &other) const;
};

// Native AS creation is deferred until the tracker's upload/build dependency pass.
class BottomLevelAccelerationStructure final {
 public:
  BottomLevelAccelerationStructure(DataUpdateTracker &tracker,
                                   graphics::BufferRange vertices,
                                   graphics::BufferRange indices,
                                   uint32_t vertex_count,
                                   uint32_t stride,
                                   uint32_t primitive_count,
                                   graphics::RayTracingGeometryFlag flags);
  ~BottomLevelAccelerationStructure();
  BottomLevelAccelerationStructure(const BottomLevelAccelerationStructure &) = delete;
  BottomLevelAccelerationStructure &operator=(const BottomLevelAccelerationStructure &) = delete;
  graphics::AccelerationStructure *Get() const;

  bool IsTrackedBy(const DataUpdateTracker &tracker) const {
    return tracker_ == &tracker;
  }

  bool HasValidInputs() const {
    return tracker_ && vertices_.buffer && indices_.buffer;
  }

  uint64_t Generation() const {
    return generation_;
  }

  AccelerationStructureInstance MakeInstance(
      const glm::mat4x3 &transform,
      uint32_t instance_id = 0,
      uint32_t instance_mask = 255,
      uint32_t hit_group_offset = 0,
      graphics::RayTracingInstanceFlag flags = graphics::RAYTRACING_INSTANCE_FLAG_NONE);

 private:
  // Only submission and lifecycle callbacks may mutate tracker-owned state.
  friend void DataUpdateTracker::Flush();
  friend void DataUpdateTracker::Unregister(Buffer *buffer);
  friend void DataUpdateTracker::Unregister(BottomLevelAccelerationStructure *blas);

  // Lifecycle and dependency notifications are restricted to the tracker.
  void DetachTracker() {
    tracker_ = nullptr;
  }

  void InvalidateBuffer(graphics::Buffer *buffer);
  void Build();
  DataUpdateTracker *tracker_;
  graphics::BufferRange vertices_, indices_;
  uint32_t vertex_count_, stride_, primitive_count_;
  graphics::RayTracingGeometryFlag flags_;
  uint64_t vertex_revision_{}, index_revision_{}, generation_{};
  std::unique_ptr<graphics::AccelerationStructure> native_;
};

class TopLevelAccelerationStructure final {
 public:
  TopLevelAccelerationStructure(DataUpdateTracker &tracker,
                                const std::vector<AccelerationStructureInstance> &instances);
  ~TopLevelAccelerationStructure();
  TopLevelAccelerationStructure(const TopLevelAccelerationStructure &) = delete;
  TopLevelAccelerationStructure &operator=(const TopLevelAccelerationStructure &) = delete;
  graphics::AccelerationStructure *Get() const;

  bool IsTrackedBy(const DataUpdateTracker &tracker) const {
    return tracker_ == &tracker;
  }

  void UpdateInstances(const std::vector<AccelerationStructureInstance> &instances);

 private:
  // Only submission and lifecycle callbacks may mutate tracker-owned state.
  friend void DataUpdateTracker::Flush();
  friend void DataUpdateTracker::Unregister(BottomLevelAccelerationStructure *blas);
  friend void DataUpdateTracker::Unregister(TopLevelAccelerationStructure *tlas);

  void DetachTracker() {
    tracker_ = nullptr;
  }

  void InvalidateBLAS(BottomLevelAccelerationStructure *blas);
  void Build();
  DataUpdateTracker *tracker_;
  bool dirty_{true};
  std::vector<AccelerationStructureInstance> instances_;
  std::vector<uint64_t> generations_;
  std::unique_ptr<graphics::AccelerationStructure> native_;
};
}  // namespace sparkium
