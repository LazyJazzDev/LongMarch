#pragma once
#include "snow_mount/solver/solver_element.h"
#include "snow_mount/solver/solver_object_pack.h"
#include "snow_mount/solver/solver_rigid_object.h"
#include "snow_mount/solver/solver_scene.h"
#include "snow_mount/solver/solver_util.h"

namespace XS::solver {

void PyBind(pybind11::module_ &m);

}
