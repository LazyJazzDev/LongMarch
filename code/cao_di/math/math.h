#pragma once
#include "cao_di/math/math_aabb.h"
#include "cao_di/math/math_basics.h"
#include "cao_di/math/math_ccd.h"
#include "cao_di/math/math_mesh.h"
#include "cao_di/math/math_mesh_sdf.h"
#include "cao_di/math/math_polynomial.h"
#include "cao_di/math/math_ray.h"
#include "cao_di/math/math_spd_projection.h"
#include "cao_di/math/math_static_collision.h"
#include "cao_di/math/math_svd.h"
#include "cao_di/math/math_triangle.h"
#include "cao_di/math/math_util.h"

namespace CD {
void PyBindMath(pybind11::module_ &m);
}  // namespace CD
