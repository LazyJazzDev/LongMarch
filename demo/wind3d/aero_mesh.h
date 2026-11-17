#pragma once
#include "aero_mesh_view.h"

namespace wind3d::aero {
// Immutable local-space geometry. Upload these arrays once per mesh; controls
// change query transforms rather than rebuilding triangle acceleration data.
class MeshGeometry {
 public:
  static std::shared_ptr<const MeshGeometry> LoadObj(const std::string &path, std::string &error);

  MeshView View() const {
    return {triangles_.data(), nodes_.data(), int(nodes_.size())};
  }

  const std::vector<MeshTriangle> &Triangles() const {
    return triangles_;
  }

  const std::vector<MeshNode> &Nodes() const {
    return nodes_;
  }

 private:
  std::vector<MeshTriangle> triangles_;
  std::vector<MeshNode> nodes_;
};

class ModelGeometry {
 public:
  static std::unique_ptr<ModelGeometry> Load(const Model &model, std::string &error);
  // Poses must come from the matching model. On failure output is unchanged.
  bool Distance(Vector point_body, const std::vector<PartPose> &poses, double &distance, std::string &error) const;

  const std::vector<std::shared_ptr<const MeshGeometry>> &Meshes() const {
    return meshes_;
  }

 private:
  std::vector<std::shared_ptr<const MeshGeometry>> meshes_;
};
}  // namespace wind3d::aero
