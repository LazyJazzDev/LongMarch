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

namespace {
struct PBDStepHelper {
  PBDSolver::RigidEntity &entity;
  Vector3<float> x_new;
  Quaternion<float> q_new;
  Vector3<float> delta_x;
  Vector3<float> delta_theta;
  int num_contacts{0};

  PBDStepHelper(PBDSolver::RigidEntity &e)
      : entity(e), x_new(e.x_), q_new(e.q_), delta_x(Vector3<float>::Zero()), delta_theta(Vector3<float>::Zero()) {
  }
};
}  // namespace

void PBDSolver::Step(float dt) {
  Vector3<float> gravity{0.0f, -9.81f, 0.0f};

  std::vector<PBDStepHelper> step_helper_;

  for (auto &[id, entity] : rigid_entities_) {
    if (entity.mass_) {
      // Semi-implicit Euler integration
      entity.v_ += dt * gravity;
    }

    step_helper_.emplace_back(entity);
    auto &helper = step_helper_.back();

    helper.x_new = entity.x_ + dt * entity.v_;

    if (entity.w_.norm() > 1e-6f) {
      Eigen::AngleAxis<float> angle_axis(dt * entity.w_.norm(), entity.w_.normalized());
      Quaternion<float> delta_q{angle_axis};
      helper.q_new = (delta_q * entity.q_).normalized();
    }
  }

  for (int step = 0; step < 20; step++) {
    for (auto &helper : step_helper_) {
      helper.delta_x = Vector3<float>::Zero();
      helper.delta_theta = Vector3<float>::Zero();
      helper.num_contacts = 0;
    }

    for (int i = 0; i < step_helper_.size(); i++) {
      MeshSDFRef mesh_sdf = step_helper_[i].entity.mesh_sdf;
      Matrix<float, 3, 3> R = step_helper_[i].q_new.toRotationMatrix();
      Vector3<float> t = step_helper_[i].x_new;
      AABB aabb_A;
      for (int k = 0; k < step_helper_[i].entity.mesh.NumVertices(); k++) {
        Vector3<float> p_A = step_helper_[i].entity.mesh.Positions()[k];
        Vector3<float> r_A = step_helper_[i].q_new * p_A + step_helper_[i].x_new;
        aabb_A.Expand(r_A);
      }

      auto &helper_A = step_helper_[i];
      for (int j = 0; j < step_helper_.size(); j++) {
        if (j == i) {
          continue;
        }

        auto &helper_B = step_helper_[j];

        for (int k = 0; k < helper_B.entity.mesh.NumVertices(); k++) {
          Vector3<float> p_B = helper_B.entity.mesh.Positions()[k];
          Vector3<float> r_B = helper_B.q_new * p_B + helper_B.x_new;
          if (!aabb_A.Contain(r_B)) {
            continue;
          }
          float sdf;
          Vector3<float> jacobian;
          mesh_sdf.SDF(r_B, R, t, &sdf, &jacobian, nullptr);
          if (sdf < 0.0f) {
            Vector3<float> r_A = r_B - sdf * jacobian;
            float C = -sdf;
            const Vector3<float> &n = jacobian;
            Vector3<float> grad_theta_A = (r_A - helper_A.x_new).cross(n);
            Vector3<float> grad_theta_B = (r_B - helper_B.x_new).cross(-n);
            float denom = n.dot(helper_A.entity.inv_mass_ * n) + n.dot(helper_B.entity.inv_mass_ * n) +
                          grad_theta_A.dot(helper_A.entity.inv_inertia_ * grad_theta_A) +
                          grad_theta_B.dot(helper_B.entity.inv_inertia_ * grad_theta_B);
            float coeff = -C / denom;
            helper_A.num_contacts++;
            helper_B.num_contacts++;
            helper_A.delta_x += coeff * helper_A.entity.inv_mass_ * n;
            helper_A.delta_theta += coeff * helper_A.entity.inv_inertia_ * grad_theta_A;
            helper_B.delta_x += coeff * helper_B.entity.inv_mass_ * -n;
            helper_B.delta_theta += coeff * helper_B.entity.inv_inertia_ * grad_theta_B;
          }
        }
      }
    }

    for (auto &helper : step_helper_) {
      if (helper.num_contacts > 0) {
        helper.x_new += helper.delta_x / helper.num_contacts;
        if (helper.delta_theta.norm() > 1e-6f) {
          Eigen::AngleAxis<float> angle_axis(helper.delta_theta.norm() / helper.num_contacts,
                                             helper.delta_theta.normalized());
          Quaternion<float> delta_q{angle_axis};
          helper.q_new = (delta_q * helper.q_new).normalized();
        }
      }
    }
  }

  for (auto &helper : step_helper_) {
    helper.entity.v_ = (helper.x_new - helper.entity.x_) / dt;
    auto delta_q = helper.q_new * helper.entity.q_.conjugate();
    Eigen::AngleAxis<float> angle_axis(delta_q);
    helper.entity.w_ = angle_axis.axis() * angle_axis.angle() / dt;
    helper.entity.x_ = helper.x_new;
    helper.entity.q_ = helper.q_new;
  }
}

}  // namespace contradium
