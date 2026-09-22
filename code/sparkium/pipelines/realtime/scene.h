#pragma once
#include "sparkium/pipelines/common/core/scene.h"

namespace sparkium::realtime {

// Owns realtime scheduling and scene invalidation. The registration backend is
// reused for material graphs, light sampling and software BVH construction.
class Scene : public render_shared::Scene {
 public:
  explicit Scene(sparkium::Scene &scene);
  void Render(sparkium::Camera *camera, sparkium::Film *film);

 private:
  uint64_t RealtimeKey() const;
  uint64_t realtime_key_{};
  bool rendered_{};
};

}  // namespace sparkium::realtime
