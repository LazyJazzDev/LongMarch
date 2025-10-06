#include "practium/model/model_mesh.h"

#include "practium/core/core.h"

namespace practium {

ModelMesh::ModelMesh(Core *core, const Mesh<float> &mesh, sparkium::Material *material)
    : ModelMesh(core, mesh, mesh, material) {
}

ModelMesh::ModelMesh(Core *core,
                     const Mesh<float> &mesh,
                     const Mesh<float> &collision_mesh,
                     sparkium::Material *material)
    : Model(core), mesh_(mesh), collision_mesh_(collision_mesh), material_(material) {
  collision_mesh_.MakeCollisionMesh();
}

sparkium::Material *ModelMesh::VisualMaterial() {
  return material_;
}

Mesh<> ModelMesh::VisualMesh() {
  return mesh_;
}

Mesh<> ModelMesh::CollisionMesh() {
  return collision_mesh_;
}

}  // namespace practium
