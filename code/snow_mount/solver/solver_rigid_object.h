#pragma once
#include "snow_mount/solver/solver_util.h"

namespace snow_mount::solver {

struct RigidObjectRef {
  MeshSDFRef mesh_sdf;
  RigidObjectState state;
  float stiffness;
  float friction;
};

struct RigidObject {
  MeshSDF mesh_sdf;
  RigidObjectState state;
  float stiffness;
  float friction;

  static void PyBind(pybind11::module_ &m);

  operator RigidObjectRef() const;
};

#if defined(__CUDACC__)
struct RigidObjectDevice {
  MeshSDFDevice mesh_sdf;
  RigidObjectState state;
  float stiffness;
  float friction;

  operator RigidObjectRef() const;
};
#endif

}  // namespace snow_mount::solver
