#include "grassland/graphics/backend/metal/metal_acceleration_structure.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_buffer.h"
#include "grassland/graphics/backend/metal/metal_core.h"
#include "grassland/graphics/frame_profile.h"

namespace grassland::graphics::backend {

void MetalAccelerationStructure::Build(MTL::AccelerationStructureDescriptor *descriptor) {
  MetalPool pool;
  if (!core_->DeviceRayQuerySupport())
    throw std::runtime_error("Metal device does not support native ray queries");
  // Complete previous users before replacing storage. The initial implementation builds
  // synchronously and skips unchanged TLAS updates; refit/compaction are separate work.
  core_->WaitGPU();
  auto sizes = core_->Device()->accelerationStructureSizes(descriptor);
  auto replacement = NS::TransferPtr(core_->Device()->newAccelerationStructure(sizes.accelerationStructureSize));
  auto scratch = NS::TransferPtr(
      core_->Device()->newBuffer(std::max<size_t>(sizes.buildScratchBufferSize, 256), MTL::ResourceStorageModePrivate));
  MetalCheck(replacement.get(), nullptr, "allocate acceleration structure");
  MetalCheck(scratch.get(), nullptr, "allocate AS scratch");
  auto command = core_->Queue()->commandBuffer();
  auto encoder = command->accelerationStructureCommandEncoder();
  encoder->buildAccelerationStructure(replacement.get(), descriptor, scratch.get(), 0);
  encoder->endEncoding();
  command->commit();
  command->waitUntilCompleted();
  if (command->status() == MTL::CommandBufferStatusError)
    MetalCheck(nullptr, command->error(), "build acceleration structure");
  structure_ = std::move(replacement);
}

MetalAccelerationStructure::MetalAccelerationStructure(MetalCore *core,
                                                       BufferRange vertices,
                                                       BufferRange indices,
                                                       uint32_t vertex_count,
                                                       uint32_t stride,
                                                       uint32_t triangle_count,
                                                       RayTracingGeometryFlag flags)
    : core_(core) {
  MetalPool pool;
  auto vertex = dynamic_cast<MetalBuffer *>(vertices.buffer);
  auto index = dynamic_cast<MetalBuffer *>(indices.buffer);
  if (!vertex || !index || vertex_count == 0 || triangle_count == 0 || stride < 12 || stride % 4 ||
      vertices.offset % 4 || indices.offset % 4 || vertices.offset > vertex->Size() || indices.offset > index->Size() ||
      vertices.size > vertex->Size() - vertices.offset || indices.size > index->Size() - indices.offset ||
      uint64_t(vertex_count - 1) * stride + 12 > vertices.size || uint64_t(triangle_count) * 12 > indices.size)
    throw std::invalid_argument("invalid Metal triangle AS buffer ranges");
  graphics::CpuProfileScope profile("native_blas_build");
  auto geometry = MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor();
  geometry->setVertexBuffer(vertex->Handle());
  geometry->setVertexBufferOffset(vertices.offset);
  geometry->setVertexStride(stride);
  geometry->setVertexFormat(MTL::AttributeFormatFloat3);
  geometry->setIndexBuffer(index->Handle());
  geometry->setIndexBufferOffset(indices.offset);
  geometry->setIndexType(MTL::IndexTypeUInt32);
  geometry->setTriangleCount(triangle_count);
  geometry->setOpaque(flags & RAYTRACING_GEOMETRY_FLAG_OPAQUE);
  geometry->setAllowDuplicateIntersectionFunctionInvocation(
      !(flags & RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION));
  auto descriptor = MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
  const NS::Object *geometries[] = {geometry};
  descriptor->setGeometryDescriptors(NS::Array::array(geometries, 1));
  Build(descriptor);
  if (graphics::FrameProfile::active)
    ++graphics::FrameProfile::active->counters["native_blas_builds"];
}

MetalAccelerationStructure::MetalAccelerationStructure(MetalCore *core,
                                                       const std::vector<RayTracingInstance> &instances)
    : core_(core),
      top_level_(true) {
  UpdateInstances(instances);
}

int MetalAccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &instances) {
  if (!top_level_)
    throw std::invalid_argument("cannot update instances on a Metal BLAS");
  MetalPool pool;
  std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> descriptors;
  std::vector<NS::SharedPtr<MTL::AccelerationStructure>> children;
  for (const auto &instance : instances) {
    auto blas = dynamic_cast<MetalAccelerationStructure *>(instance.acceleration_structure);
    if (!blas || blas->top_level_ || blas->core_ != core_)
      throw std::invalid_argument("Metal TLAS requires BLAS from the same graphics core");
    auto child =
        std::find_if(children.begin(), children.end(), [&](const auto &p) { return p.get() == blas->Handle(); });
    auto index = static_cast<uint32_t>(child - children.begin());
    if (child == children.end())
      children.push_back(NS::RetainPtr(blas->Handle()));
    MTL::AccelerationStructureUserIDInstanceDescriptor descriptor{};
    glm::mat4 transform(1.0f);
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 3; ++r) {
        auto value = instance.transform[r][c];
        if (!std::isfinite(value))
          throw std::invalid_argument("Metal instance transform must be finite");
        descriptor.transformationMatrix.columns[c][r] = transform[c][r] = value;
      }
    if (std::abs(glm::determinant(transform)) < 1e-20f)
      throw std::invalid_argument("Metal instance transform must be invertible");
    auto flags = instance.instance_flags;
    if ((flags & RAYTRACING_INSTANCE_FLAG_OPAQUE) && (flags & RAYTRACING_INSTANCE_FLAG_NO_OPAQUE))
      throw std::invalid_argument("conflicting Metal instance opacity flags");
    descriptor.options = (flags & RAYTRACING_INSTANCE_FLAG_TRIANGLE_FLIP_FACING)
                             ? MTL::AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise
                             : MTL::AccelerationStructureInstanceOptionNone;
    if (flags & RAYTRACING_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE)
      descriptor.options |= MTL::AccelerationStructureInstanceOptionDisableTriangleCulling;
    if (flags & RAYTRACING_INSTANCE_FLAG_OPAQUE)
      descriptor.options |= MTL::AccelerationStructureInstanceOptionOpaque;
    if (flags & RAYTRACING_INSTANCE_FLAG_NO_OPAQUE)
      descriptor.options |= MTL::AccelerationStructureInstanceOptionNonOpaque;
    descriptor.mask = instance.instance_mask;
    descriptor.userID = instance.instance_id;
    descriptor.intersectionFunctionTableOffset = instance.instance_hit_group_offset;
    descriptor.accelerationStructureIndex = index;
    descriptors.push_back(descriptor);
  }

  bool same = structure_ && descriptors.size() == instances_.size() && children.size() == children_.size();
  if (same && !descriptors.empty())
    same = std::memcmp(descriptors.data(), instances_.data(), descriptors.size() * sizeof(descriptors[0])) == 0;
  for (size_t i = 0; same && i < children.size(); ++i)
    same = children[i].get() == children_[i].get();
  if (same)
    return 0;
  graphics::CpuProfileScope profile("native_tlas_build");
  auto buffer = NS::TransferPtr(core_->Device()->newBuffer(
      std::max<size_t>(1, descriptors.size()) * sizeof(descriptors[0]), MTL::ResourceStorageModeShared));
  MetalCheck(buffer.get(), nullptr, "instance descriptor buffer");
  if (!descriptors.empty())
    std::memcpy(buffer->contents(), descriptors.data(), descriptors.size() * sizeof(descriptors[0]));
  auto descriptor = MTL::InstanceAccelerationStructureDescriptor::descriptor();
  descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
  descriptor->setInstanceDescriptorStride(sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor));
  descriptor->setInstanceDescriptorBuffer(buffer.get());
  descriptor->setInstanceCount(descriptors.size());
  std::vector<const NS::Object *> objects;
  for (const auto &child : children)
    objects.push_back(child.get());
  descriptor->setInstancedAccelerationStructures(NS::Array::array(objects.data(), objects.size()));
  Build(descriptor);
  instances_ = std::move(descriptors);
  children_ = std::move(children);
  if (graphics::FrameProfile::active)
    ++graphics::FrameProfile::active->counters["native_tlas_builds"];
  return 0;
}
}  // namespace grassland::graphics::backend
