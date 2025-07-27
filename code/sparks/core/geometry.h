#pragma once
#include "sparks/core/core_util.h"

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

class Geometry {
 public:
  Geometry(Core *core, const Mesh<float> &mesh);

  graphics::Buffer *Buffer();
  graphics::AccelerationStructure *BLAS();
  graphics::HitGroup HitGroup();

 private:
  Core *core_;
  std::unique_ptr<graphics::Buffer> geometry_buffer_;
  std::unique_ptr<graphics::AccelerationStructure> blas_;
  std::unique_ptr<graphics::Shader> closest_hit_shader_;
};

}  // namespace sparks
