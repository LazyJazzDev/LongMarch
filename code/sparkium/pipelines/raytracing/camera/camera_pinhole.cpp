#include "sparkium/pipelines/raytracing/camera/camera_pinhole.h"

#include "sparkium/pipelines/raytracing/core/core.h"

namespace sparkium::raytracing {

CameraPinhole::CameraPinhole(sparkium::CameraPinhole &camera)
    : Camera(DedicatedCast(camera.GetCore()), "camera/pinhole.slang", "CameraPinhole"),
      camera_(camera) {
  buffer_ = camera_.GetCore()->GetDevice()->CreateBuffer({sizeof(CameraPinholeData)});
}

rhi::IBuffer *CameraPinhole::Buffer() {
  CameraPinholeData data{};
  data.world_to_camera = camera_.view;
  data.camera_to_world = glm::inverse(camera_.view);
  data.scale = glm::vec2(camera_.aspect * tan(camera_.fovy * 0.5f), tan(camera_.fovy * 0.5f));
  UploadBuffer(camera_.GetCore()->GetDevice(), buffer_.Get(), &data, sizeof(data));
  return buffer_.Get();
}

}  // namespace sparkium::raytracing
