#pragma once
#include "xue_shan/solver/solver_element.h"
#include "xue_shan/solver/solver_object_pack.h"
#include "xue_shan/solver/solver_rigid_object.h"
#include "xue_shan/solver/solver_scene.h"
#include "xue_shan/solver/solver_util.h"

namespace XS::solver {

void PyBind(pybind11::module_ &m);

}
