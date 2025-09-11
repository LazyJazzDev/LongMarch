#pragma once
#include "snowberg/solver/solver_element.h"
#include "snowberg/solver/solver_object_pack.h"
#include "snowberg/solver/solver_rigid_object.h"
#include "snowberg/solver/solver_util.h"

#if defined(__CUDACC__)
#include "thrust/device_vector.h"
#endif

namespace snowberg::solver {

struct SceneRef {
  int num_particle;
  Vector3<float> *x_prev;
  Vector3<float> *x;
  Vector3<float> *v;
  float *m;
  int *particle_ids;

  int num_stretching;
  ElementStretching *stretchings;
  int *stretching_indices;
  int *stretching_ids;
  DirectoryRef stretching_directory;

  int num_bending;
  ElementBending *bendings;
  int *bending_indices;
  int *bending_ids;
  DirectoryRef bending_directory;

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

  int ParticleIndex(int particle_id) const;
  int StretchingIndex(int stretching_id) const;
  int BendingIndex(int bending_id) const;
  int RigidObjectIndex(int rigid_object_id) const;

  std::vector<Vector3<float>> GetPositions(const std::vector<int> &particle_ids) const;

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
  SceneDevice(const Scene &scene);
  ~SceneDevice();

  std::vector<Vector3<float>> GetPositions(const std::vector<int> &particle_ids) const;

  RigidObjectState GetRigidObjectState(int rigid_object_id) const;
  void SetRigidObjectState(int rigid_object_id, const RigidObjectState &state);

  float GetRigidObjectStiffness(int rigid_object_id) const;
  void SetRigidObjectStiffness(int rigid_object_id, float stiffness);

  float GetRigidObjectFriction(int rigid_object_id) const;
  void SetRigidObjectFriction(int rigid_object_id, float friction);

  operator SceneRef();

  static void Update(SceneDevice &scene, float dt);

  static void UpdateBatch(const std::vector<SceneDevice *> &scenes, float dt);

 private:
  thrust::device_vector<Vector3<float>> x_prev_;
  thrust::device_vector<Vector3<float>> x_;
  thrust::device_vector<Vector3<float>> v_;
  thrust::device_vector<float> m_;
  thrust::device_vector<int> particle_ids_;
  std::vector<int> particle_ids_host_;
  int next_particle_id_{0};
  thrust::device_vector<int> particle_colors_;
  DirectoryDevice particle_directory_;
  Directory particle_directory_host_;

  thrust::device_vector<ElementStretching> stretchings_;
  thrust::device_vector<int> stretching_indices_;
  thrust::device_vector<int> stretching_ids_;
  std::vector<int> stretching_ids_host_;
  int next_stretching_id_{0};
  DirectoryDevice stretching_directory_;

  thrust::device_vector<ElementBending> bendings_;
  thrust::device_vector<int> bending_indices_;
  thrust::device_vector<int> bending_ids_;
  std::vector<int> bending_ids_host_;
  int next_bending_id_{0};
  DirectoryDevice bending_directory_;

  thrust::device_vector<RigidObjectRef> rigid_objects_;
  std::vector<MeshSDFDevice> rigid_object_meshes_;
  thrust::device_vector<int> rigid_object_ids_;
  std::vector<int> rigid_object_ids_host_;
  int next_rigid_object_id_{0};

  cudaStream_t stream_;
};
#endif

}  // namespace snowberg::solver
