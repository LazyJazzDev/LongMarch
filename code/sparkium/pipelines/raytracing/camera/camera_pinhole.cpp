#include "sparkium/pipelines/raytracing/camera/camera_pinhole.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

CameraPinhole::CameraPinhole(sparkium::CameraPinhole &camera)
    : Camera(DedicatedCast(camera.GetCore()), "camera/pinhole.slang", "CameraPinhole"),
      camera_(camera) {
  DedicatedCast(camera.GetCore())
      ->GraphicsCore()
      ->CreateBuffer(sizeof(CameraPinholeData), graphics::BUFFER_TYPE_STATIC, &buffer_);
}

graphics::Buffer *CameraPinhole::Buffer() {
  CameraPinholeData data{};
  data.world_to_camera = camera_.view;
  data.camera_to_world = glm::inverse(camera_.view);
  data.scale = glm::vec2(camera_.aspect * tan(camera_.fovy * 0.5f), tan(camera_.fovy * 0.5f));
  buffer_->UploadData(&data, sizeof(data));
  return buffer_.get();
}

}  // namespace sparkium::raytracing
