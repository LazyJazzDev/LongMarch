#pragma once
#include "practium/model/model_util.h"

namespace practium {

class ModelMesh : public Model {
 public:
  ModelMesh(Core *core, const Mesh<float> &mesh, sparkium::Material *material);
  ModelMesh(Core *core, const Mesh<float> &mesh, const Mesh<float> &collision_mesh, sparkium::Material *material);
  sparkium::Material *VisualMaterial() override;
  Mesh<> VisualMesh() override;
  Mesh<> CollisionMesh() override;

  Mesh<> mesh_;
  Mesh<> collision_mesh_;
  sparkium::Material *material_;
};

}  // namespace practium
