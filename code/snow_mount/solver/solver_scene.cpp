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
    stretching_indices_.push_back(object_pack_view.particle_ids[object_pack.stretching_indices[i * 3]]);
    stretching_indices_.push_back(object_pack_view.particle_ids[object_pack.stretching_indices[i * 3 + 1]]);
    stretching_indices_.push_back(object_pack_view.particle_ids[object_pack.stretching_indices[i * 3 + 2]]);
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
    bending_indices_.push_back(object_pack_view.particle_ids[object_pack.bending_indices[i * 4]]);
    bending_indices_.push_back(object_pack_view.particle_ids[object_pack.bending_indices[i * 4 + 1]]);
    bending_indices_.push_back(object_pack_view.particle_ids[object_pack.bending_indices[i * 4 + 2]]);
    bending_indices_.push_back(object_pack_view.particle_ids[object_pack.bending_indices[i * 4 + 3]]);
  }

  return object_pack_view;
}

int Scene::AddRigidBody(const RigidObject &rigid_object) {
  int rigid_object_id = next_rigid_object_id_++;
  rigid_object_meshes_.push_back(rigid_object.mesh_sdf);
  rigid_object_ids_.push_back(rigid_object_id);
  RigidObjectRef rigid_object_ref = rigid_object;
  rigid_object_ref.mesh_sdf = rigid_object_meshes_.back();
  rigid_objects_.push_back(rigid_object_ref);
  return rigid_object_id;
}

int Scene::ParticleIndex(int particle_id) const {
  return BinarySearch(particle_ids_.data(), particle_ids_.size(), particle_id);
}

int Scene::StretchingIndex(int stretching_id) const {
  return BinarySearch(stretching_ids_.data(), stretchings_.size(), stretching_id);
}

int Scene::BendingIndex(int bending_id) const {
  return BinarySearch(bending_ids_.data(), bendings_.size(), bending_id);
}

int Scene::RigidObjectIndex(int rigid_object_id) const {
  return BinarySearch(rigid_object_ids_.data(), rigid_objects_.size(), rigid_object_id);
}

std::vector<Vector3<float>> Scene::GetPositions(const std::vector<int> &particle_ids) const {
  std::vector<Vector3<float>> positions;
  for (auto id : particle_ids) {
    positions.push_back(x_[ParticleIndex(id)]);
  }
  return positions;
}

void Scene::PyBind(pybind11::module_ &m) {
  pybind11::class_<Scene, std::shared_ptr<Scene>> scene(m, "Scene");
  scene.def(pybind11::init<>());
  scene.def("add_object", &Scene::AddObject);
  scene.def("add_rigid_object", &Scene::AddRigidBody);
  scene.def("get_positions", &Scene::GetPositions);

#if defined(__CUDACC__)
  pybind11::class_<SceneDevice, std::shared_ptr<SceneDevice>> scene_device(m, "SceneDevice");
  scene_device.def(pybind11::init<const Scene &>(), pybind11::arg("scene"));
  scene_device.def("get_positions", &SceneDevice::GetPositions);
  scene_device.def("get_rigid_object_state", &SceneDevice::GetRigidObjectState, pybind11::arg("rigid_object_id"));
  scene_device.def("set_rigid_object_state", &SceneDevice::SetRigidObjectState, pybind11::arg("rigid_object_id"),
                   pybind11::arg("state"));
  scene_device.def("get_rigid_object_stiffness", &SceneDevice::GetRigidObjectStiffness,
                   pybind11::arg("rigid_object_id"));
  scene_device.def("set_rigid_object_stiffness", &SceneDevice::SetRigidObjectStiffness,
                   pybind11::arg("rigid_object_id"), pybind11::arg("stiffness"));
  scene_device.def("get_rigid_object_friction", &SceneDevice::GetRigidObjectFriction, pybind11::arg("rigid_object_id"));
  scene_device.def("set_rigid_object_friction", &SceneDevice::SetRigidObjectFriction, pybind11::arg("rigid_object_id"),
                   pybind11::arg("friction"));
  m.def("update_scene", &SceneDevice::Update);
  m.def("update_scene_batch", &SceneDevice::UpdateBatch);
#endif
}

#if defined(__CUDACC__)
SceneDevice::SceneDevice(const Scene &scene) {
  x_prev_ = scene.x_;
  x_ = scene.x_;
  v_ = scene.v_;
  m_ = scene.m_;
  particle_ids_ = scene.particle_ids_;
  particle_ids_host_ = scene.particle_ids_;
  next_particle_id_ = scene.next_particle_id_;

  stretchings_ = scene.stretchings_;
  stretching_indices_ = scene.stretching_indices_;
  stretching_ids_ = scene.stretching_ids_;
  stretching_ids_host_ = scene.stretching_ids_;
  next_stretching_id_ = scene.next_stretching_id_;

  bendings_ = scene.bendings_;
  bending_indices_ = scene.bending_indices_;
  bending_ids_ = scene.bending_ids_;
  bending_ids_host_ = scene.bending_ids_;
  next_bending_id_ = scene.next_bending_id_;

  std::vector<RigidObjectRef> rigid_objects = scene.rigid_objects_;
  rigid_object_meshes_.reserve(scene.rigid_object_meshes_.size());
  for (int i = 0; i < scene.rigid_object_meshes_.size(); i++) {
    rigid_object_meshes_.emplace_back(scene.rigid_object_meshes_[i]);
    rigid_objects[i].mesh_sdf = rigid_object_meshes_.back();
  }
  rigid_objects_ = rigid_objects;
  rigid_object_ids_ = scene.rigid_object_ids_;
  rigid_object_ids_host_ = scene.rigid_object_ids_;
  next_rigid_object_id_ = scene.next_rigid_object_id_;

  int num_particle = x_.size();
  Directory stretching_directory{scene.stretching_indices_, num_particle};
  stretching_directory_ = stretching_directory;
  Directory bending_directory{scene.bending_indices_, num_particle};
  bending_directory_ = bending_directory;

  std::vector<int> particle_colors(num_particle, -1);

  int c = 0;
  for (int colored = 0; colored < num_particle; c++) {
    for (int i = 0; i < num_particle; i++) {
      if (particle_colors[i] != -1)
        continue;
      int id = scene.particle_ids_[i];
      bool pass = true;
      auto check_conflict = [&](int other_id) {
        if (other_id == id)
          return;
        int idx = BinarySearch(scene.particle_ids_.data(), num_particle, other_id);
        if (particle_colors[idx] == c)
          pass = false;
      };

      // printf("idx: %d, id: %d\n", i, id);

      for (int j = 0, first = stretching_directory.first[id], size = stretching_directory.count[id]; pass && j < size;
           j++) {
        // printf("stretching - j: %d, first: %d, size: %d count: %d\n", j, first, size,
        // stretching_directory.positions.size());
        int stretching_i = stretching_directory.positions[first + j] / 3;
        check_conflict(scene.stretching_indices_[stretching_i * 3]);
        check_conflict(scene.stretching_indices_[stretching_i * 3 + 1]);
        check_conflict(scene.stretching_indices_[stretching_i * 3 + 2]);
      }

      for (int j = 0, first = bending_directory.first[id], size = bending_directory.count[id]; pass && j < size; j++) {
        // printf("bending - j: %d, first: %d, size: %d count: %d position: %d\n", j, first, size,
        // bending_directory.positions.size(), bending_directory.positions[first + j]);
        int bending_i = bending_directory.positions[first + j] / 4;
        // printf("%d %d %d %d %d\n", bending_i * 4, bending_i * 4 + 1, bending_i * 4 + 2, bending_i * 4 + 3,
        // bending_directory.positions.size()); printf("%d %d %d %d\n", scene.bending_indices_[bending_i * 4],
        // scene.bending_indices_[bending_i * 4 + 1], scene.bending_indices_[bending_i * 4 + 2],
        // scene.bending_indices_[bending_i * 4 + 3]);
        check_conflict(scene.bending_indices_[bending_i * 4]);
        // puts("check0");
        check_conflict(scene.bending_indices_[bending_i * 4 + 1]);
        // puts("check1");
        check_conflict(scene.bending_indices_[bending_i * 4 + 2]);
        // puts("check2");
        check_conflict(scene.bending_indices_[bending_i * 4 + 3]);
        // puts("check3");
      }

      if (pass) {
        particle_colors[i] = c;
        colored++;
      }
    }
  }
  // puts("here");
  particle_colors_ = particle_colors;
  particle_directory_host_ = Directory(particle_colors, c);
  particle_directory_ = particle_directory_host_;

  cudaStreamCreate(&stream_);
}

SceneDevice::~SceneDevice() {
  cudaStreamDestroy(stream_);
}

namespace {
struct SearchAndCopyOp {
  const Vector3<float> *x;
  const int *particle_ids;
  int num_particle;

  LM_DEVICE_FUNC Vector3<float> operator()(int particle_id) {
    return x[BinarySearch(particle_ids, num_particle, particle_id)];
  }
};
}  // namespace

std::vector<Vector3<float>> SceneDevice::GetPositions(const std::vector<int> &particle_ids) const {
  thrust::device_vector<int> query_particle_ids(particle_ids);
  thrust::device_vector<Vector3<float>> result_positions(particle_ids.size());
  thrust::transform(query_particle_ids.begin(), query_particle_ids.end(), result_positions.begin(),
                    SearchAndCopyOp{x_.data().get(), particle_ids_.data().get(), static_cast<int>(x_.size())});
  std::vector<Vector3<float>> result_positions_host(particle_ids.size());
  thrust::copy(result_positions.begin(), result_positions.end(), result_positions_host.begin());
  return result_positions_host;
}

RigidObjectState SceneDevice::GetRigidObjectState(int rigid_object_id) const {
  int rigid_object_idx = BinarySearch(rigid_object_ids_host_.data(), rigid_object_ids_host_.size(), rigid_object_id);
  RigidObjectRef ref = rigid_objects_[rigid_object_idx];
  return ref.state;
}

void SceneDevice::SetRigidObjectState(int rigid_object_id, const RigidObjectState &state) {
  int rigid_object_idx = BinarySearch(rigid_object_ids_host_.data(), rigid_object_ids_host_.size(), rigid_object_id);
  RigidObjectRef ref = rigid_objects_[rigid_object_idx];
  ref.state = state;
  rigid_objects_[rigid_object_idx] = ref;
}

float SceneDevice::GetRigidObjectStiffness(int rigid_object_id) const {
  int rigid_object_idx = BinarySearch(rigid_object_ids_host_.data(), rigid_object_ids_host_.size(), rigid_object_id);
  RigidObjectRef ref = rigid_objects_[rigid_object_idx];
  return ref.stiffness;
}

void SceneDevice::SetRigidObjectStiffness(int rigid_object_id, float stiffness) {
  int rigid_object_idx = BinarySearch(rigid_object_ids_host_.data(), rigid_object_ids_host_.size(), rigid_object_id);
  RigidObjectRef ref = rigid_objects_[rigid_object_idx];
  ref.stiffness = stiffness;
  rigid_objects_[rigid_object_idx] = ref;
}

float SceneDevice::GetRigidObjectFriction(int rigid_object_id) const {
  int rigid_object_idx = BinarySearch(rigid_object_ids_host_.data(), rigid_object_ids_host_.size(), rigid_object_id);
  RigidObjectRef ref = rigid_objects_[rigid_object_idx];
  return ref.friction;
}

void SceneDevice::SetRigidObjectFriction(int rigid_object_id, float friction) {
  int rigid_object_idx = BinarySearch(rigid_object_ids_host_.data(), rigid_object_ids_host_.size(), rigid_object_id);
  RigidObjectRef ref = rigid_objects_[rigid_object_idx];
  ref.friction = friction;
  rigid_objects_[rigid_object_idx] = ref;
}

SceneDevice::operator SceneRef() {
  SceneRef scene_ref{};

  scene_ref.num_particle = x_.size();
  scene_ref.x_prev = thrust::raw_pointer_cast(x_prev_.data());
  scene_ref.x = thrust::raw_pointer_cast(x_.data());
  scene_ref.v = thrust::raw_pointer_cast(v_.data());
  scene_ref.m = thrust::raw_pointer_cast(m_.data());
  scene_ref.particle_ids = thrust::raw_pointer_cast(particle_ids_.data());

  scene_ref.num_stretching = stretchings_.size();
  scene_ref.stretchings = thrust::raw_pointer_cast(stretchings_.data());
  scene_ref.stretching_indices = thrust::raw_pointer_cast(stretching_indices_.data());
  scene_ref.stretching_ids = thrust::raw_pointer_cast(stretching_ids_.data());
  scene_ref.stretching_directory = stretching_directory_;

  scene_ref.num_bending = bendings_.size();
  scene_ref.bendings = thrust::raw_pointer_cast(bendings_.data());
  scene_ref.bending_indices = thrust::raw_pointer_cast(bending_indices_.data());
  scene_ref.bending_ids = thrust::raw_pointer_cast(bending_ids_.data());
  scene_ref.bending_directory = bending_directory_;

  scene_ref.num_rigid_object = rigid_objects_.size();
  scene_ref.rigid_objects = thrust::raw_pointer_cast(rigid_objects_.data());
  scene_ref.rigid_object_ids = thrust::raw_pointer_cast(rigid_object_ids_.data());

  return scene_ref;
}
#endif

}  // namespace snow_mount::solver
