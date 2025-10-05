#include "contradium/pbd/pbd_solver.h"

namespace contradium {

int PBDSolver::AddEntity(const Mesh<float> &mesh,
                         const Vector3<float> &x,
                         const Quaternion<float> &q,
                         float mass,
                         float inertia) {
  RigidEntity entity;
  entity.mesh = mesh;
  entity.mesh_sdf = MeshSDF{VertexBufferView{mesh.Positions()}, mesh.NumVertices(), mesh.Indices(), mesh.NumIndices()};
  entity.x_ = x;
  entity.q_ = q;
  entity.mass_ = mass;
  entity.inertia_ = inertia;
  if (entity.mass_) {
    entity.inv_mass_ = 1.0f / entity.mass_;
  } else {
    entity.inv_mass_ = 0.0f;
  }

  if (entity.inertia_) {
    entity.inv_inertia_ = 1.0f / entity.inertia_;
  } else {
    entity.inv_inertia_ = 0.0f;
  }

  entity.v_ = Vector3<float>{0.0f, 0.0f, 0.0f};
  entity.w_ = Vector3<float>{0.0f, 0.0f, 0.0f};

  int entity_id = rigid_entity_id_counter_++;
  rigid_entities_[entity_id] = entity;
  return entity_id;
}

}  // namespace contradium
