#pragma once
#include "sparkium/core/geometry.h"

namespace sparkium {

class GeometryMesh : public Geometry {
 public:
  struct Header {
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

  GeometryMesh(Core *core, const Mesh<float> &mesh);

  int PrimitiveCount() override;

 private:
  Header header_;
  std::unique_ptr<graphics::Buffer> geometry_buffer_;
  int primitive_count_;
};

}  // namespace sparkium
