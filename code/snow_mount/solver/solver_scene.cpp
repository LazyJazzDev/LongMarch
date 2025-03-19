#include "snow_mount/solver/solver_scene.h"

namespace snow_mount::solver {
LM_DEVICE_FUNC int SceneRef::ParticleIndex(int particle_id) const {
  return BinarySearch(particle_ids, num_particle, particle_id);
}

LM_DEVICE_FUNC int SceneRef::StretchingIndex(int stretching_id) const {
  return BinarySearch(stretching_ids, num_stretching, stretching_id);
}

LM_DEVICE_FUNC int SceneRef::BendingIndex(int bending_id) const {
  return BinarySearch(bending_ids, num_bending, bending_id);
}

LM_DEVICE_FUNC int SceneRef::RigidObjectIndex(int rigid_object_id) const {
  return BinarySearch(rigid_object_ids, num_rigid_object, rigid_object_id);
}

ObjectPackView Scene::AddObject(const ObjectPack &object_pack) {
  ObjectPackView object_pack_view;

  int pack_num_particle = object_pack.x.size();
  int first_particle_id = next_particle_id_;
  next_particle_id_ += pack_num_particle;
  object_pack_view.particle_ids.reserve(pack_num_particle);
  for (int i = 0; i < pack_num_particle; i++) {
    int particle_id = first_particle_id + i;
    object_pack_view.particle_ids.push_back(particle_id);
    particle_ids_.push_back(particle_id);
    x_.push_back(object_pack.x[i]);
    v_.push_back(object_pack.v[i]);
    m_.push_back(object_pack.m[i]);
  }

  int pack_num_stretching = object_pack.stretchings.size();
  int first_stretching_id = next_stretching_id_;
  next_stretching_id_ += pack_num_stretching;
  object_pack_view.stretching_ids.reserve(pack_num_stretching);
  for (int i = 0; i < pack_num_stretching; i++) {
    int stretching_id = first_stretching_id + i;
    object_pack_view.stretching_ids.push_back(stretching_id);
    stretching_ids_.push_back(stretching_id);
    stretchings_.push_back(object_pack.stretchings[i]);
    stretching_indices_.push_back(object_pack_view.particle_ids[object_pack.stretching_indices[i]]);
  }

  int pack_num_bending = object_pack.bendings.size();
  int first_bending_id = next_bending_id_;
  next_bending_id_ += pack_num_bending;
  object_pack_view.bending_ids.reserve(pack_num_bending);
  for (int i = 0; i < pack_num_bending; i++) {
    int bending_id = first_bending_id + i;
    object_pack_view.bending_ids.push_back(bending_id);
    bending_ids_.push_back(bending_id);
    bendings_.push_back(object_pack.bendings[i]);
    bending_indices_.push_back(object_pack_view.particle_ids[object_pack.bending_indices[i]]);
  }

  return object_pack_view;
}

int Scene::AddRigidBody(const RigidObject &rigid_object) {
  int rigid_object_id = next_rigid_object_id_++;
  rigid_object_meshes_.push_back(rigid_object.mesh_sdf);
  rigid_object_ids_.push_back(rigid_object_id);
  RigidObjectRef rigid_object_ref;
  rigid_object_ref.mesh_sdf = rigid_object_meshes_.back();
  rigid_object_ref.mesh_sdf.rotation = rigid_object.R;
  rigid_object_ref.mesh_sdf.translation = rigid_object.t;
  rigid_object_ref.v = rigid_object.v;
  rigid_object_ref.omega = rigid_object.omega;
  rigid_object_ref.mass = rigid_object.mass;
  rigid_object_ref.inertia = rigid_object.inertia;
  rigid_object_ref.stiffness = rigid_object.stiffness;
  return rigid_object_id;
}

LM_DEVICE_FUNC int Scene::ParticleIndex(int particle_id) const {
  return BinarySearch(particle_ids_.data(), particle_ids_.size(), particle_id);
}

LM_DEVICE_FUNC int Scene::StretchingIndex(int stretching_id) const {
  return BinarySearch(stretching_ids_.data(), stretchings_.size(), stretching_id);
}

LM_DEVICE_FUNC int Scene::BendingIndex(int bending_id) const {
  return BinarySearch(bending_ids_.data(), bendings_.size(), bending_id);
}

LM_DEVICE_FUNC int Scene::RigidObjectIndex(int rigid_object_id) const {
  return BinarySearch(rigid_object_ids_.data(), rigid_objects_.size(), rigid_object_id);
}

std::vector<Vector3<float>> Scene::GetPositions(const std::vector<int> &particle_ids) const {
  std::vector<Vector3<float>> positions;
  for (auto id : particle_ids) {
    positions.push_back(x_[ParticleIndex(id)]);
  }
  return positions;
}

Scene::operator SceneRef() {
  SceneRef scene_ref;

  scene_ref.num_particle = x_.size();
  scene_ref.x = x_.data();
  scene_ref.v = v_.data();
  scene_ref.m = m_.data();
  scene_ref.particle_ids = particle_ids_.data();

  scene_ref.num_stretching = stretchings_.size();
  scene_ref.stretchings = stretchings_.data();
  scene_ref.stretching_indices = stretching_indices_.data();
  scene_ref.stretching_ids = stretching_ids_.data();

  scene_ref.num_bending = bendings_.size();
  scene_ref.bendings = bendings_.data();
  scene_ref.bending_indices = bending_indices_.data();
  scene_ref.bending_ids = bending_ids_.data();

  scene_ref.num_rigid_object = rigid_objects_.size();
  scene_ref.rigid_objects = rigid_objects_.data();
  scene_ref.rigid_object_ids = rigid_object_ids_.data();

  return scene_ref;
}

void Scene::PyBind(pybind11::module_ &m) {
  pybind11::class_<Scene, std::shared_ptr<Scene>> scene(m, "Scene");
  scene.def(pybind11::init<>());
  scene.def("add_object", &Scene::AddObject);
  scene.def("get_positions", &Scene::GetPositions);
}

}  // namespace snow_mount::solver
