#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

struct CameraData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 view_proj;
  glm::mat4 inv_view;
  glm::mat4 inv_proj;
  glm::mat4 inv_view_proj;
};

class Camera : public Object {
 public:
  explicit Camera(sparkium::Camera &camera);
  graphics::Buffer *NearFieldBuffer() const {
    return near_field_buffer_.get();
  }
  void Update();

 private:
  sparkium::Camera &camera_;
  Core *core_{};
  std::unique_ptr<graphics::Buffer> near_field_buffer_;
  std::unique_ptr<graphics::Buffer> far_field_buffer_;
  CameraData data_{};
};

Camera *DedicatedCast(sparkium::Camera *camera);

}  // namespace sparkium::raster
