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
    rhi::BindingLayoutDesc layout;
    layout.bindings = {{0, rhi::BINDING_UNIFORM_BUFFER}};
    near_field_bindings_ = core_->GetDevice()->CreateBindingGroup(layout);
    far_field_bindings_ = core_->GetDevice()->CreateBindingGroup(layout);
    if (!near_field_buffer_ || !far_field_buffer_ || !near_field_bindings_ || !far_field_bindings_ ||
        !near_field_bindings_->BindResource(0, near_field_buffer_.Get()) ||
        !far_field_bindings_->BindResource(0, far_field_buffer_.Get())) {
      core_->ReportFailure("Failed to create raster camera resources");
      return;
    }
    Update();
  }
}

rhi::IBindingGroup *Camera::NearFieldBindings() const {
  return near_field_bindings_.Get();
}

rhi::IBindingGroup *Camera::FarFieldBindings() const {
  return far_field_bindings_.Get();
}

void Camera::Update() {
  data_.view = camera_.view;
  data_.proj = glm::perspectiveZO(camera_.fovy, camera_.aspect, 0.1f, 100.0f);
  data_.view_proj = data_.proj * data_.view;
  data_.inv_view = glm::inverse(data_.view);
  data_.inv_proj = glm::inverse(data_.proj);
  data_.inv_view_proj = glm::inverse(data_.view_proj);
  core_->QueueUpload(near_field_buffer_.Get(), &data_, sizeof(CameraData));
  data_.proj = glm::perspectiveZO(camera_.fovy, camera_.aspect, 100.0f, 10000.0f);
  data_.view_proj = data_.proj * data_.view;
  data_.inv_proj = glm::inverse(data_.proj);
  data_.inv_view_proj = glm::inverse(data_.view_proj);
  core_->QueueUpload(far_field_buffer_.Get(), &data_, sizeof(CameraData));
}

Camera *DedicatedCast(sparkium::Camera *camera) {
  COMPONENT_CAST(camera, Camera);
}

}  // namespace sparkium::raster
