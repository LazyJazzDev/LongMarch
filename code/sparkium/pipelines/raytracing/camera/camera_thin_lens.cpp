#include "sparkium/pipelines/raytracing/camera/camera_thin_lens.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

CameraThinLens::CameraThinLens(sparkium::CameraThinLens &camera)
    : Camera(DedicatedCast(camera.GetCore()), "camera/thin_lens.slang", "CameraThinLens"),
      camera_(camera) {
  DedicatedCast(camera.GetCore())
      ->GraphicsCore()
      ->CreateBuffer(sizeof(CameraThinLensData), graphics::BUFFER_TYPE_STATIC, &buffer_);
}

graphics::Buffer *CameraThinLens::Buffer() {
  CameraThinLensData data{};
  data.world_to_camera = camera_.view;
  data.camera_to_world = glm::inverse(camera_.view);
  data.scale = glm::vec2(camera_.aspect * tan(camera_.fovy * 0.5f), tan(camera_.fovy * 0.5f));
  data.aperture_radius = camera_.aperture_radius;
  data.focus_distance = camera_.focus_distance;
  data.aperture_blades = camera_.aperture_blades;
  data.aperture_rotation = camera_.aperture_rotation;
  data.aperture_ratio = camera_.aperture_ratio;
  buffer_->UploadData(&data, sizeof(data));
  return buffer_.get();
}

}  // namespace sparkium::raytracing
