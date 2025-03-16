#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {

struct Camera {
  glm::mat4 proj;
  glm::mat4 view;

  static Matrix4<float> LookAt(const Vector3<float> &eye, const Vector3<float> &center, const Vector3<float> &up);
  static Matrix4<float> Perspective(float fovy, float aspect, float z_near, float z_far);
  static void PyBind(pybind11::module_ &m);
};

}  // namespace snow_mount::visualizer
