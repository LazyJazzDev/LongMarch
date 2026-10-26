#include "sparkium/pipelines/raster/core/camera.h"

#include <glm/ext/matrix_clip_space.hpp>

#include "sparkium/core/camera.h"
#include "sparkium/pipelines/raster/core/core.h"

namespace sparkium::raster {

Camera::Camera(sparkium::Camera &camera) : camera_(camera) {
  core_ = DedicatedCast(camera.GetCore());
  if (core_) {
    near_field_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(CameraData) + 255) & ~uint64_t(255)});
    far_field_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(CameraData) + 255) & ~uint64_t(255)});
    Update();
  }
}

rhi::IBuffer *Camera::NearFieldBuffer() const {
  return near_field_buffer_.Get();
}

rhi::IBuffer *Camera::FarFieldBuffer() const {
  return far_field_buffer_.Get();
}

void Camera::Update() {
  data_.view = camera_.view;
  data_.proj = glm::perspectiveZO(camera_.fovy, camera_.aspect, 0.1f, 100.0f);
  data_.view_proj = data_.proj * data_.view;
  data_.inv_view = glm::inverse(data_.view);
  data_.inv_proj = glm::inverse(data_.proj);
  data_.inv_view_proj = glm::inverse(data_.view_proj);
  UploadBuffer(core_->GetDevice(), near_field_buffer_.Get(), &data_, sizeof(CameraData));
  data_.proj = glm::perspectiveZO(camera_.fovy, camera_.aspect, 100.0f, 10000.0f);
  data_.view_proj = data_.proj * data_.view;
  data_.inv_proj = glm::inverse(data_.proj);
  data_.inv_view_proj = glm::inverse(data_.view_proj);
  UploadBuffer(core_->GetDevice(), far_field_buffer_.Get(), &data_, sizeof(CameraData));
}

Camera *DedicatedCast(sparkium::Camera *camera) {
  COMPONENT_CAST(camera, Camera);
}

}  // namespace sparkium::raster
