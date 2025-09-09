#pragma once
#include "snowberg/solver/solver_element.h"
#include "snowberg/solver/solver_object_pack.h"
#include "snowberg/solver/solver_rigid_object.h"
#include "snowberg/solver/solver_scene.h"
#include "snowberg/solver/solver_util.h"

namespace snowberg::solver {

void PyBind(pybind11::module_ &m);

}
