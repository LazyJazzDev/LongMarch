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
    uint32_t color_offset;
    uint32_t color_stride;
  };

  GeometryMesh(Core *core, const Mesh<float> &mesh);
  GeometryMesh(Core *core, std::shared_ptr<const Mesh<float>> mesh);

  const Mesh<float> *HostMesh() const {
    return host_mesh_.get();
  }

  int PrimitiveCount() override;
  graphics::Buffer *GetBuffer() const;
  const Header &GetHeader() const;

 private:
  Header header_{};
  std::shared_ptr<const Mesh<float>> host_mesh_;
  std::unique_ptr<graphics::Buffer> geometry_buffer_;
  int primitive_count_;
};

}  // namespace sparkium
