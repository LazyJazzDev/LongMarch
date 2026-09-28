#include "sparkium/core/acceleration_structure.h"

#include <stdexcept>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
bool AccelerationStructureInstance::operator==(const AccelerationStructureInstance &other) const {
  return blas == other.blas && transform == other.transform && instance_id == other.instance_id &&
         instance_mask == other.instance_mask && instance_hit_group_offset == other.instance_hit_group_offset &&
         instance_flags == other.instance_flags;
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(DataUpdateTracker &tracker,
                                                                   graphics::BufferRange vertices,
                                                                   graphics::BufferRange indices,
                                                                   uint32_t vertex_count,
                                                                   uint32_t stride,
                                                                   uint32_t primitive_count,
                                                                   graphics::RayTracingGeometryFlag flags)
    : tracker_(&tracker),
      vertices_(vertices),
      indices_(indices),
      vertex_count_(vertex_count),
      stride_(stride),
      primitive_count_(primitive_count),
      flags_(flags) {
  tracker.Revision(vertices.buffer);
  tracker.Revision(indices.buffer);
  tracker.Register(this);
}

BottomLevelAccelerationStructure::~BottomLevelAccelerationStructure() {
  if (tracker_)
    tracker_->Unregister(this);
}

graphics::AccelerationStructure *BottomLevelAccelerationStructure::Get() const {
  if (!tracker_)
    throw std::logic_error("acceleration structure tracker was destroyed");
  if (!vertices_.buffer || !indices_.buffer)
    throw std::logic_error("BLAS input buffer was destroyed");
  return native_.get();
}

AccelerationStructureInstance BottomLevelAccelerationStructure::MakeInstance(const glm::mat4x3 &transform,
                                                                             uint32_t instance_id,
                                                                             uint32_t instance_mask,
                                                                             uint32_t hit_group_offset,
                                                                             graphics::RayTracingInstanceFlag flags) {
  return {this, transform, instance_id, instance_mask, hit_group_offset, flags};
}

void BottomLevelAccelerationStructure::InvalidateBuffer(graphics::Buffer *buffer) {
  if (vertices_.buffer == buffer)
    vertices_.buffer = nullptr;
  if (indices_.buffer == buffer)
    indices_.buffer = nullptr;
}

void BottomLevelAccelerationStructure::Build() {
  if (!vertices_.buffer || !indices_.buffer)
    return;
  auto vertex_revision = tracker_->Revision(vertices_.buffer);
  auto index_revision = tracker_->Revision(indices_.buffer);
  if (native_ && vertex_revision == vertex_revision_ && index_revision == index_revision_)
    return;
  auto valid_range = [](graphics::BufferRange range) {
    return range.offset <= range.buffer->Size() && range.size <= range.buffer->Size() - range.offset;
  };
  if (!valid_range(vertices_) || !valid_range(indices_) || !vertex_count_ || stride_ < 12 ||
      uint64_t(vertex_count_ - 1) * stride_ + 12 > vertices_.size ||
      uint64_t(primitive_count_) * 3 * sizeof(uint32_t) > indices_.size)
    throw std::out_of_range("tracked BLAS geometry exceeds its input buffers");
  std::unique_ptr<graphics::AccelerationStructure> replacement;
  if (tracker_->GetCore()->CreateBottomLevelAccelerationStructure(vertices_, indices_, vertex_count_, stride_,
                                                                  primitive_count_, flags_, &replacement))
    throw std::runtime_error("failed to build tracked BLAS");
  native_ = std::move(replacement);
  vertex_revision_ = vertex_revision;
  index_revision_ = index_revision;
  ++generation_;
  if (graphics::FrameProfile::active)
    ++graphics::FrameProfile::active->counters["data_update_blas_builds"];
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(
    DataUpdateTracker &tracker,
    const std::vector<AccelerationStructureInstance> &instances)
    : tracker_(&tracker) {
  UpdateInstances(instances);
  tracker.Register(this);
}

TopLevelAccelerationStructure::~TopLevelAccelerationStructure() {
  if (tracker_)
    tracker_->Unregister(this);
}

graphics::AccelerationStructure *TopLevelAccelerationStructure::Get() const {
  if (!tracker_)
    throw std::logic_error("acceleration structure tracker was destroyed");
  for (const auto &instance : instances_) {
    if (!instance.blas)
      throw std::logic_error("TLAS input BLAS was destroyed");
    instance.blas->Get();
  }
  return native_.get();
}

void TopLevelAccelerationStructure::UpdateInstances(const std::vector<AccelerationStructureInstance> &instances) {
  if (!tracker_)
    throw std::logic_error("acceleration structure's DataUpdateTracker has been destroyed");
  for (const auto &instance : instances)
    if (!tracker_->Contains(instance.blas))
      throw std::invalid_argument("TLAS requires BLAS resources from the same tracker");
  if (instances_ != instances) {
    instances_ = instances;
    dirty_ = true;
  }
}

void TopLevelAccelerationStructure::InvalidateBLAS(BottomLevelAccelerationStructure *blas) {
  for (auto &instance : instances_)
    if (instance.blas == blas) {
      instance.blas = nullptr;
      dirty_ = true;
    }
}

void TopLevelAccelerationStructure::Build() {
  bool changed = !native_ || dirty_ || generations_.size() != instances_.size();
  for (size_t i = 0; i < instances_.size(); ++i) {
    auto *blas = instances_[i].blas;
    if (!blas || !blas->HasValidInputs())
      return;
    if (!blas->Get())
      throw std::logic_error("TLAS requires a built BLAS");
    if (!changed && generations_[i] != blas->Generation())
      changed = true;
  }
  if (!changed)
    return;
  std::vector<uint64_t> generations;
  std::vector<graphics::RayTracingInstance> native_instances;
  generations.reserve(instances_.size());
  native_instances.reserve(instances_.size());
  for (const auto &instance : instances_) {
    generations.push_back(instance.blas->Generation());
    native_instances.push_back(
        instance.blas->Get()->MakeInstance(instance.transform, instance.instance_id, instance.instance_mask,
                                           instance.instance_hit_group_offset, instance.instance_flags));
  }
  const bool updating = native_ != nullptr;
  int result = updating ? native_->UpdateInstances(native_instances)
                        : tracker_->GetCore()->CreateTopLevelAccelerationStructure(native_instances, &native_);
  if (result)
    throw std::runtime_error("failed to build tracked TLAS");
  generations_ = std::move(generations);
  dirty_ = false;
  if (graphics::FrameProfile::active)
    ++graphics::FrameProfile::active->counters[updating ? "data_update_tlas_updates" : "data_update_tlas_builds"];
}
}  // namespace sparkium
