#include "snow_mount/solver/solver_rigid_object.h"

namespace snow_mount::solver {

void RigidObject::PyBind(pybind11::module_ &m) {
  pybind11::class_<RigidObject> rigid_object(m, "RigidObject");
  rigid_object.def(pybind11::init<MeshSDF, Matrix3<float>, Vector3<float>, Vector3<float>, Vector3<float>, float,
                                  Matrix3<float>, float>(),
                   pybind11::arg("mesh_sdf"), pybind11::arg("R") = Matrix3<float>::Identity(),
                   pybind11::arg("t") = Vector3<float>::Zero(), pybind11::arg("v") = Vector3<float>::Zero(),
                   pybind11::arg("omega") = Vector3<float>::Zero(), pybind11::arg("mass") = 1.0,
                   pybind11::arg("inertia") = Matrix3<float>::Identity(), pybind11::arg("stiffness") = 1e5f);
  rigid_object.def_readwrite("mesh_sdf", &RigidObject::mesh_sdf);
  rigid_object.def_readwrite("state", &RigidObject::state);
  rigid_object.def_readwrite("stiffness", &RigidObject::stiffness);

  pybind11::class_<RigidObjectState> rigid_object_state(m, "RigidObjectState");
  rigid_object_state.def(
      pybind11::init<Matrix3<float>, Vector3<float>, Vector3<float>, Vector3<float>, float, Matrix3<float>>(),
      pybind11::arg("R") = Matrix3<float>::Identity(), pybind11::arg("t") = Vector3<float>::Zero(),
      pybind11::arg("v") = Vector3<float>::Zero(), pybind11::arg("omega") = Vector3<float>::Zero(),
      pybind11::arg("mass") = 1.0, pybind11::arg("inertia") = Matrix3<float>::Identity());
  rigid_object_state.def("__repr__", [](const RigidObjectState &state) {
    return fmt::format("RigidObjectState(\nR={},\nt={},\nv={},\nomega={},\nmass={},\ninertia={}\n)", state.R, state.t,
                       state.v, state.omega, state.mass, state.inertia);
  });
  rigid_object_state.def_readwrite("R", &RigidObjectState::R);
  rigid_object_state.def_readwrite("t", &RigidObjectState::t);
  rigid_object_state.def_readwrite("v", &RigidObjectState::v);
  rigid_object_state.def_readwrite("omega", &RigidObjectState::omega);
  rigid_object_state.def_readwrite("mass", &RigidObjectState::mass);
  rigid_object_state.def_readwrite("inertia", &RigidObjectState::inertia);
  rigid_object_state.def("next_state", &RigidObjectState::NextState);
}

RigidObject::operator RigidObjectRef() const {
  RigidObjectRef rigid_object;
  rigid_object.mesh_sdf = mesh_sdf;
  rigid_object.state = state;
  rigid_object.stiffness = stiffness;
  return rigid_object;
}

RigidObjectDevice::operator RigidObjectRef() const {
  RigidObjectRef rigid_object;
  rigid_object.mesh_sdf = mesh_sdf;
  rigid_object.state = state;
  rigid_object.stiffness = stiffness;
  return rigid_object;
}
}  // namespace snow_mount::solver
