#include "snowberg/visualizer/visualizer_camera.h"

#include "glm/gtc/matrix_transform.hpp"
#include "snowberg/visualizer/visualizer_core.h"

namespace snowberg::visualizer {

Camera::Camera(const std::shared_ptr<Core> &core) : core_(core) {
  proj = glm::mat4{1.0f};
  view = glm::mat4{1.0f};
  core_->GraphicsCore()->CreateBuffer(sizeof(CameraInfo), graphics::BUFFER_TYPE_DYNAMIC, &camera_buffer_);
}

std::shared_ptr<Core> Camera::GetCore() const {
  return core_;
}

CameraInfo Camera::GetInfo() const {
  return {proj, view};
}

void Camera::SetInfo(const CameraInfo &camera_info) {
  proj = camera_info.proj;
  view = camera_info.view;
}

Matrix4<float> Camera::LookAt(const Vector3<float> &eye, const Vector3<float> &center, const Vector3<float> &up) {
  return GLMToEigen(glm::lookAt(EigenToGLM(eye), EigenToGLM(center), EigenToGLM(up)));
}

Matrix4<float> Camera::Perspective(float fovy, float aspect, float z_near, float z_far) {
  return GLMToEigen(glm::perspective(fovy, aspect, z_near, z_far));
}

}  // namespace snowberg::visualizer
