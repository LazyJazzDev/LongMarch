#pragma once
#include "snow_mount/solver/solver_util.h"

namespace snow_mount::solver {

struct RigidObjectRef {
  MeshSDFRef mesh_sdf;
  Matrix3<float> R;
  Vector3<float> t;
  Vector3<float> v;
  Vector3<float> omega;
  float mass;
  Matrix3<float> inertia;
  float stiffness;
};

struct RigidObject {
  MeshSDF mesh_sdf;
  Matrix3<float> R;
  Vector3<float> t;
  Vector3<float> v;
  Vector3<float> omega;
  float mass;
  Matrix3<float> inertia;
  float stiffness;

  static void PyBind(pybind11::module_ &m);

  operator RigidObjectRef() const;
};

#if defined(__CUDACC__)
struct RigidObjectDevice {
  MeshSDFDevice mesh_sdf;
  Matrix3<float> R;
  Vector3<float> t;
  Vector3<float> v;
  Vector3<float> omega;
  float mass;
  Matrix3<float> inertia;
  float stiffness;

  operator RigidObjectRef() const;
};
#endif

}  // namespace snow_mount::solver
