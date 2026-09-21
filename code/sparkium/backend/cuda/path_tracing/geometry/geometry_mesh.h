#pragma once
#include "sparkium/backend/cuda/path_tracing/core/geometry.h"

namespace sparkium::cuda_tracing {

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

  GeometryMesh(sparkium::GeometryMesh &geometry);
  const Mesh<float> *HostMesh() const;

  graphics::Buffer *Buffer() override;
  graphics::AccelerationStructure *BLAS() override;
  int PrimitiveCount() override;
  const CodeLines &SamplerImpl() const override;

 private:
  sparkium::GeometryMesh &geometry_;
  std::unique_ptr<graphics::AccelerationStructure> blas_;
  CodeLines sampler_implementation_;
};

}  // namespace sparkium::cuda_tracing
