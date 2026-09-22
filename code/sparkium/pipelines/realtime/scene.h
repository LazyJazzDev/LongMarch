#pragma once
#include "sparkium/pipelines/raytracing/core/scene.h"

namespace sparkium::realtime {

// Owns realtime scheduling and scene invalidation. The registration backend is
// reused for material graphs, light sampling and software BVH construction.
class Scene : public Object {
 public:
  explicit Scene(sparkium::Scene &scene);
  void Render(sparkium::Camera *camera, sparkium::Film *film);

 private:
  uint64_t RealtimeKey() const;
  sparkium::Scene &scene_;
  raytracing::Scene tracing_scene_;
  uint64_t realtime_key_{};
  bool rendered_{};
};

}  // namespace sparkium::realtime
