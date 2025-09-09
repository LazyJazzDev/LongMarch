#pragma once
#include "cao_di/bvh/bvh.h"
#include "cao_di/graphics/graphics.h"
#include "cao_di/math/math.h"
#include "cao_di/physics/physics.h"
#include "cao_di/util/util.h"
#include "cao_di/vulkan/vulkan.h"

#ifdef _WIN64
#include "cao_di/d3d12/direct3d12.h"
#endif  // _WIN64

namespace CD {

void PyBind(pybind11::module_ &m);

}
