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

void PBDSolver::SetPosition(int rigid_entity_id, const Vector3<float> &x) {
  rigid_entities_.at(rigid_entity_id).x_ = x;
}

void PBDSolver::SetOrientation(int rigid_entity_id, const Quaternion<float> &q) {
  rigid_entities_.at(rigid_entity_id).q_ = q;
}

void PBDSolver::SetMass(int rigid_entity_id, float mass) {
  rigid_entities_.at(rigid_entity_id).mass_ = mass;
  if (mass) {
    rigid_entities_.at(rigid_entity_id).inv_mass_ = 1.0f / mass;
  } else {
    rigid_entities_.at(rigid_entity_id).inv_mass_ = 0.0f;
  }
}

void PBDSolver::SetInertia(int rigid_entity_id, float inertia) {
  rigid_entities_.at(rigid_entity_id).inertia_ = inertia;
  if (inertia) {
    rigid_entities_.at(rigid_entity_id).inv_inertia_ = 1.0f / inertia;
  } else {
    rigid_entities_.at(rigid_entity_id).inv_inertia_ = 0.0f;
  }
}

void PBDSolver::SetVelocity(int rigid_entity_id, const Vector3<float> &v) {
  rigid_entities_.at(rigid_entity_id).v_ = v;
}

void PBDSolver::SetAngularVelocity(int rigid_entity_id, const Vector3<float> &w) {
  rigid_entities_.at(rigid_entity_id).w_ = w;
}

const PBDSolver::RigidEntity &PBDSolver::GetEntity(int rigid_entity_id) const {
  return rigid_entities_.at(rigid_entity_id);
}

void PBDSolver::Step(float dt) {
  Vector3<float> gravity{0.0f, -9.81f, 0.0f};
  for (auto &[id, entity] : rigid_entities_) {
    if (entity.mass_) {
      // Semi-implicit Euler integration
      entity.v_ += dt * gravity;
      entity.x_ += dt * entity.v_;

      if (entity.w_.norm() > 0.0f) {
        Eigen::AngleAxis<float> angle_axis(dt * entity.w_.norm(), entity.w_.normalized());
        Quaternion<float> delta_q{angle_axis};
        entity.q_ = (delta_q * entity.q_).normalized();
      }
    }
  }
}

}  // namespace contradium
