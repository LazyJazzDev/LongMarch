#pragma once
#include "snow_mount/solver/solver_util.h"

namespace snow_mount::solver {

struct ElementStretching {
  float mu{0.0f};
  float lambda{0.0f};
  float area{0.0f};
  float damping{0.0f};
  Matrix2<float> Dm{Matrix2<float>::Identity()};
  float sigma_lb{-1.0f};
  float sigma_ub{-1.0f};
};

struct ElementBending {
  float stiffness{0.0f};
  float damping{0.0f};
  float theta_rest{0.0f};
  float bending_bound{4.0f};  // any value larger than pi
};

namespace element {
void PyBind(pybind11::module_ &m);
}

}  // namespace snow_mount::solver
