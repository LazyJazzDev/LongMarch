#include "snow_mount/visualizer/visualizer_camera.h"

#include "glm/gtc/matrix_transform.hpp"

namespace snow_mount::visualizer {

Matrix4<float> Camera::LookAt(const Vector3<float> &eye, const Vector3<float> &center, const Vector3<float> &up) {
  return GLMToEigen(glm::lookAt(EigenToGLM(eye), EigenToGLM(center), EigenToGLM(up)));
}

Matrix4<float> Camera::Perspective(float fovy, float aspect, float z_near, float z_far) {
  return GLMToEigen(glm::perspective(fovy, aspect, z_near, z_far));
}

void Camera::PyBind(pybind11::module_ &m) {
  pybind11::class_<Camera> camera(m, "Camera");
  camera.def(pybind11::init([](const Matrix4<float> &proj, const Matrix4<float> &view) {
               Camera cam;
               cam.proj = EigenToGLM(proj);
               cam.view = EigenToGLM(view);
               return cam;
             }),
             pybind11::arg("proj") = Matrix4<float>::Identity(), pybind11::arg("view") = Matrix4<float>::Identity());
  camera.def("__repr__", [](const Camera &camera) {
    return pybind11::str("Camera(proj=\n{},\nview=\n{})").format(GLMToEigen(camera.proj), GLMToEigen(camera.view));
  });
  camera.def_property(
      "proj", [](Camera &cam) { return GLMToEigen(cam.proj); },
      [](Camera &cam, const Matrix4<float> &proj) { cam.proj = EigenToGLM(proj); });
  camera.def_property(
      "view", [](Camera &cam) { return GLMToEigen(cam.view); },
      [](Camera &cam, const Matrix4<float> &view) { cam.view = EigenToGLM(view); });
  m.def("look_at", &Camera::LookAt, pybind11::arg("eye"), pybind11::arg("center"), pybind11::arg("up"));
  m.def("perspective", &Camera::Perspective, pybind11::arg("fovy"), pybind11::arg("aspect"), pybind11::arg("near"),
        pybind11::arg("far"));
}

}  // namespace snow_mount::visualizer
