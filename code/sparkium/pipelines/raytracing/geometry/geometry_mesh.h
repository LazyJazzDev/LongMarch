#pragma once
#include "sparkium/pipelines/raytracing/core/geometry.h"

namespace sparkium::raytracing {

class GeometryMesh : public Geometry {
 public:
  using Header = sparkium::GeometryMesh::Header;

  GeometryMesh(sparkium::GeometryMesh &geometry);

  graphics::Buffer *Buffer() override;
  graphics::AccelerationStructure *BLAS() override;
  const CodeLines &ClosestHitShaderImpl() const override;
  int PrimitiveCount() override;
  const CodeLines &SamplerImpl() const override;

 private:
  sparkium::GeometryMesh &geometry_;
  std::unique_ptr<graphics::AccelerationStructure> blas_;
  CodeLines sampler_implementation_;
  CodeLines closest_hit_shader_implementation_;
};

}  // namespace sparkium::raytracing
