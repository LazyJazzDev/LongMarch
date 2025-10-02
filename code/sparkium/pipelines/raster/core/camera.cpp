#include "sparkium/pipelines/raster/core/camera.h"

#include <glm/ext/matrix_clip_space.hpp>

#include "sparkium/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

Camera::Camera(sparkium::Camera &camera) : camera_(camera) {
  core_ = DedicatedCast(camera.GetCore());
  if (core_) {
    core_->GraphicsCore()->CreateBuffer(sizeof(CameraData), graphics::BUFFER_TYPE_STATIC, &near_field_buffer_);
    core_->GraphicsCore()->CreateBuffer(sizeof(CameraData), graphics::BUFFER_TYPE_STATIC, &far_field_buffer_);
    Update();
  }
}

graphics::Buffer *Camera::NearFieldBuffer() const {
  return near_field_buffer_.get();
}

graphics::Buffer *Camera::FarFieldBuffer() const {
  return far_field_buffer_.get();
}

void Camera::Update() {
  data_.view = camera_.view;
  data_.proj = glm::perspective(camera_.fovy, camera_.aspect, 0.1f, 100.0f);
  data_.view_proj = data_.proj * data_.view;
  data_.inv_view = glm::inverse(data_.view);
  data_.inv_proj = glm::inverse(data_.proj);
  data_.inv_view_proj = glm::inverse(data_.view_proj);
  near_field_buffer_->UploadData(&data_, sizeof(CameraData));
  data_.proj = glm::perspective(camera_.fovy, camera_.aspect, 100.0f, 10000.0f);
  data_.view_proj = data_.proj * data_.view;
  data_.inv_proj = glm::inverse(data_.proj);
  data_.inv_view_proj = glm::inverse(data_.view_proj);
  far_field_buffer_->UploadData(&data_, sizeof(CameraData));
}

Camera *DedicatedCast(sparkium::Camera *camera) {
  COMPONENT_CAST(camera, Camera);
}

}  // namespace sparkium::raster
