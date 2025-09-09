#pragma once
#include "xue_shan/visualizer/visualizer_util.h"

namespace XS::visualizer {

class Camera {
  friend class Core;
  Camera(const std::shared_ptr<Core> &core);

 public:
  glm::mat4 proj;
  glm::mat4 view;

  std::shared_ptr<Core> GetCore() const;

  CameraInfo GetInfo() const;
  void SetInfo(const CameraInfo &camera_info);

  static Matrix4<float> LookAt(const Vector3<float> &eye, const Vector3<float> &center, const Vector3<float> &up);
  static Matrix4<float> Perspective(float fovy, float aspect, float z_near, float z_far);
  static void PyBind(pybind11::module_ &m);

 private:
  std::shared_ptr<Core> core_;
  std::unique_ptr<graphics::Buffer> camera_buffer_;
};

}  // namespace XS::visualizer
