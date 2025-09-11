#include "snowberg/solver/solver_rigid_object.h"

namespace snowberg::solver {

RigidObject::operator RigidObjectRef() const {
  RigidObjectRef rigid_object;
  rigid_object.mesh_sdf = mesh_sdf;
  rigid_object.state = state;
  rigid_object.stiffness = stiffness;
  rigid_object.friction = friction;
  return rigid_object;
}

#if defined(__CUDACC__)
RigidObjectDevice::operator RigidObjectRef() const {
  RigidObjectRef rigid_object;
  rigid_object.mesh_sdf = mesh_sdf;
  rigid_object.state = state;
  rigid_object.stiffness = stiffness;
  rigid_object.friction = friction;
  return rigid_object;
}
#endif
}  // namespace snowberg::solver
