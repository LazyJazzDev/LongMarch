#include "snow_mount/solver/solver_rigid_object.h"

namespace snow_mount::solver {
RigidObject::operator RigidObjectRef() const {
  RigidObjectRef rigid_object;
  rigid_object.mesh_sdf = mesh_sdf;
  rigid_object.mesh_sdf.rotation = R;
  rigid_object.mesh_sdf.translation = t;
  rigid_object.v = v;
  rigid_object.omega = omega;
  rigid_object.mass = mass;
  rigid_object.inertia = inertia;
  rigid_object.stiffness = stiffness;
  return rigid_object;
}

RigidObjectDevice::operator RigidObjectRef() const {
  RigidObjectRef rigid_object;
  rigid_object.mesh_sdf = mesh_sdf;
  rigid_object.mesh_sdf.rotation = R;
  rigid_object.mesh_sdf.translation = t;
  rigid_object.v = v;
  rigid_object.omega = omega;
  rigid_object.mass = mass;
  rigid_object.inertia = inertia;
  rigid_object.stiffness = stiffness;
  return rigid_object;
}
}  // namespace snow_mount::solver
