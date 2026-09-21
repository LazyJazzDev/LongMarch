#include "sparkium/backend/cuda/geometry.h"

namespace sparkium::backend::cuda {
Geometry::Geometry(Core *core, std::shared_ptr<const GeometryDefinition> data) : source(std::move(data)) {
  if (auto mesh = std::get_if<grassland::Mesh<float>>(&source->shape)) {
    auto shared_mesh = std::shared_ptr<const grassland::Mesh<float>>(source, mesh);
    object = std::make_unique<GeometryMesh>(core, std::move(shared_mesh));
  } else {
    const auto &hair = std::get<HairData>(source->shape);
    object = std::make_unique<GeometryHair>(core, hair.points, hair.radii, hair.offsets, hair.radial_segments);
  }
}
}  // namespace sparkium::backend::cuda
