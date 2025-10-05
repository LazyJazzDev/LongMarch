#pragma once
#include "contradium/pbd/pbd_util.h"

namespace contradium {

class PBDSolver {
 public:
  struct RigidEntity {
    Mesh<float> mesh;
    MeshSDF mesh_sdf;
    float mass_;
    float inv_mass_;
    float inertia_;
    float inv_inertia_;
    Vector3<float> x_;
    Quaternion<float> q_;
    Vector3<float> v_;
    Vector3<float> w_;
  };

  int AddEntity(const Mesh<float> &mesh,
                const Vector3<float> &x = {0.0f, 0.0f, 0.0f},
                const Quaternion<float> &q = {1.0f, 0.0f, 0.0f, 0.0f},
                float mass = 1.0f,
                float inertia = 1.0f);

  void SetPosition(int rigid_entity_id, const Vector3<float> &x);
  void SetOrientation(int rigid_entity_id, const Quaternion<float> &q);
  void SetMass(int rigid_entity_id, float mass);
  void SetInertia(int rigid_entity_id, float inertia);
  void SetVelocity(int rigid_entity_id, const Vector3<float> &v);
  void SetAngularVelocity(int rigid_entity_id, const Vector3<float> &w);
  const RigidEntity &GetEntity(int rigid_entity_id) const;
  void Step(float dt);

 private:
  std::map<int, RigidEntity> rigid_entities_;
  int rigid_entity_id_counter_{0};
};

}  // namespace contradium
