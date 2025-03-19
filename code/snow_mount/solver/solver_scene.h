#pragma once
#include "snow_mount/solver/solver_element.h"
#include "snow_mount/solver/solver_object_pack.h"
#include "snow_mount/solver/solver_rigid_object.h"
#include "snow_mount/solver/solver_util.h"

namespace snow_mount::solver {

struct SceneRef {
  int num_particle;
  Vector3<float> *x;
  Vector3<float> *v;
  float *m;
  int *particle_ids;

  int num_stretching;
  ElementStretching *stretchings;
  int *stretching_indices;
  int *stretching_ids;

  int num_bending;
  ElementBending *bendings;
  int *bending_indices;
  int *bending_ids;

  int num_rigid_object;
  RigidObjectRef *rigid_objects;
  int *rigid_object_ids;

  LM_DEVICE_FUNC int ParticleIndex(int particle_id) const;
  LM_DEVICE_FUNC int StretchingIndex(int stretching_id) const;
  LM_DEVICE_FUNC int BendingIndex(int bending_id) const;
  LM_DEVICE_FUNC int RigidObjectIndex(int rigid_object_id) const;
};

class Scene {
 public:
  ObjectPackView AddObject(const ObjectPack &object_pack);
  int AddRigidBody(const RigidObject &rigid_object);

  LM_DEVICE_FUNC int ParticleIndex(int particle_id) const;
  LM_DEVICE_FUNC int StretchingIndex(int stretching_id) const;
  LM_DEVICE_FUNC int BendingIndex(int bending_id) const;
  LM_DEVICE_FUNC int RigidObjectIndex(int rigid_object_id) const;

  LM_DEVICE_FUNC std::vector<Vector3<float>> GetPositions(const std::vector<int> &particle_ids) const;

  operator SceneRef();

  static void PyBind(pybind11::module_ &m);

 private:
  friend class SceneDevice;
  std::vector<Vector3<float>> x_;
  std::vector<Vector3<float>> v_;
  std::vector<float> m_;
  std::vector<int> particle_ids_;
  int next_particle_id_{0};

  std::vector<ElementStretching> stretchings_;
  std::vector<int> stretching_indices_;
  std::vector<int> stretching_ids_;
  int next_stretching_id_{0};

  std::vector<ElementBending> bendings_;
  std::vector<int> bending_indices_;
  std::vector<int> bending_ids_;
  int next_bending_id_{0};

  std::vector<RigidObjectRef> rigid_objects_;
  std::vector<MeshSDF> rigid_object_meshes_;
  std::vector<int> rigid_object_ids_;
  int next_rigid_object_id_{0};
};

#if defined(__CUDACC__)
class SceneDevice {
 public:
  SceneDevice() {
  }

 private:
};
#endif

}  // namespace snow_mount::solver
