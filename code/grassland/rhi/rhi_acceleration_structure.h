#pragma once
#include <array>

#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi {
// Describes fixed geometry shape, not input buffer ownership. Counts/format/stride
// are unchanged for this object's builds. Update also preserves index topology.
struct TriangleGeometryDesc {
  uint32_t vertex_count{};
  uint32_t vertex_stride{12};
  uint32_t index_count{};  // Zero selects non-indexed triangles.
  IndexFormat index_format{INDEX_FORMAT_UINT32};
  bool opaque{true};
};

struct AABBGeometryDesc {
  uint32_t count{};
  uint32_t stride{24};  // float3 min followed by float3 max.
  bool opaque{true};
};

struct BottomLevelAccelerationStructureDesc {
  // Exactly one geometry family must be nonempty.
  std::vector<TriangleGeometryDesc> geometries;
  std::vector<AABBGeometryDesc> aabbs;
  bool allow_update{};
  bool allow_compaction{};
  // Zero requests ordinary build storage. Nonzero allocates a compact-copy destination.
  uint64_t compacted_size{};
};

struct TopLevelAccelerationStructureDesc {
  // Allocated capacity. BUILD may use fewer instances; UPDATE must preserve the last build count.
  uint32_t instance_count{};
  bool allow_update{};
  bool allow_compaction{};
  uint64_t compacted_size{};
};

struct AccelerationStructureGeometryBuffers {
  BufferView data;  // Vertex positions or AABBs.
  BufferView indices;
};

class IAccelerationStructure : public IObject {
 public:
  virtual AccelerationStructureType GetType() const = 0;
  // Call only after the submission containing CmdQueryCompactedSize completes.
  // Zero indicates no available result. This query never waits for the GPU.
  virtual uint64_t GetCompactedSize() const = 0;

 protected:
  ~IAccelerationStructure() override;
};

class IBottomLevelAccelerationStructure : public IAccelerationStructure {
 public:
  virtual const BottomLevelAccelerationStructureDesc &GetDesc() const = 0;

 protected:
  ~IBottomLevelAccelerationStructure() override;
};

// TLAS retains its BLAS dependencies. Geometry buffers are only needed by builds.
class ITopLevelAccelerationStructure : public IAccelerationStructure {
 public:
  virtual const TopLevelAccelerationStructureDesc &GetDesc() const = 0;

 protected:
  ~ITopLevelAccelerationStructure() override;
};

struct RayTracingInstance {
  IBottomLevelAccelerationStructure *blas{};
  // Row-major 3x4 object-to-world matrix, independent of shader matrix layout.
  std::array<float, 12> transform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  uint32_t custom_index{};      // 24 bits.
  uint32_t hit_group_offset{};  // 24 bits, in shader table records.
  uint8_t mask{0xff};
  bool disable_triangle_culling{};
  bool front_counter_clockwise{};
};
}  // namespace grassland::rhi
