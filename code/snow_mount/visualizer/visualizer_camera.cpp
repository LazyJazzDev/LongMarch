#include "snow_mount/visualizer/visualizer_camera.h"

#include "glm/gtc/matrix_transform.hpp"
#include "snow_mount/visualizer/visualizer_core.h"

namespace snow_mount::visualizer {

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

void Camera::PyBind(pybind11::module_ &m) {
  pybind11::class_<CameraInfo> camera_info(m, "CameraInfo");
  camera_info.def(pybind11::init([](const Matrix4<float> &proj, const Matrix4<float> &view) {
                    CameraInfo cam{};
                    cam.proj = EigenToGLM(proj);
                    cam.view = EigenToGLM(view);
                    return cam;
                  }),
                  pybind11::arg("proj") = Matrix4<float>::Identity(),
                  pybind11::arg("view") = Matrix4<float>::Identity());
  camera_info.def("__repr__", [](const CameraInfo &camera) {
    return pybind11::str("CameraInfo(\nproj=\n{},\nview=\n{}\n)")
        .format(GLMToEigen(camera.proj), GLMToEigen(camera.view));
  });
  camera_info.def_property(
      "proj", [](const CameraInfo &cam) { return GLMToEigen(cam.proj); },
      [](CameraInfo &cam, const Matrix4<float> &proj) { cam.proj = EigenToGLM(proj); });
  camera_info.def_property(
      "view", [](const CameraInfo &cam) { return GLMToEigen(cam.view); },
      [](CameraInfo &cam, const Matrix4<float> &view) { cam.view = EigenToGLM(view); });
  pybind11::class_<Camera, std::shared_ptr<Camera>> camera(m, "Camera");
  camera.def("__repr__", [](const Camera &camera) {
    return pybind11::str("Camera(\nproj=\n{},\nview=\n{}\n)").format(GLMToEigen(camera.proj), GLMToEigen(camera.view));
  });
  camera.def("get_core", &Camera::GetCore);
  camera.def_property(
      "proj", [](const Camera &cam) { return GLMToEigen(cam.proj); },
      [](Camera &cam, const Matrix4<float> &proj) { cam.proj = EigenToGLM(proj); });
  camera.def_property(
      "view", [](const Camera &cam) { return GLMToEigen(cam.view); },
      [](Camera &cam, const Matrix4<float> &view) { cam.view = EigenToGLM(view); });
  camera.def("get_info", &Camera::GetInfo);
  camera.def("set_info", &Camera::SetInfo, pybind11::arg("camera_info"));

  m.def("look_at", &Camera::LookAt, pybind11::arg("eye"), pybind11::arg("center"), pybind11::arg("up"));
  m.def("perspective", &Camera::Perspective, pybind11::arg("fovy"), pybind11::arg("aspect"), pybind11::arg("near"),
        pybind11::arg("far"));
}

}  // namespace snow_mount::visualizer
