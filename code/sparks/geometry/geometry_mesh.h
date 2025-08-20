#pragma once
#include "sparks/core/geometry.h"

namespace sparks {

struct GeometryMeshHeader {
  uint32_t num_vertices;
  uint32_t num_indices;
  uint32_t position_offset;
  uint32_t position_stride;
  uint32_t normal_offset;
  uint32_t normal_stride;
  uint32_t tex_coord_offset;
  uint32_t tex_coord_stride;
  uint32_t tangent_offset;
  uint32_t tangent_stride;
  uint32_t signal_offset;
  uint32_t signal_stride;
  uint32_t index_offset;
};

class GeometryMesh : public Geometry {
 public:
  GeometryMesh(Core *core, const Mesh<float> &mesh);

  graphics::Buffer *Buffer() override;
  graphics::AccelerationStructure *BLAS() override;
  const CodeLines &ClosestHitShaderImpl() const override;
  int PrimitiveCount() override;
  const CodeLines &SamplerImpl() const override;

 private:
  std::unique_ptr<graphics::Buffer> geometry_buffer_;
  std::unique_ptr<graphics::AccelerationStructure> blas_;
  int primitive_count_;
  CodeLines sampler_implementation_;
  CodeLines closest_hit_shader_implementation_;
};

}  // namespace sparks
