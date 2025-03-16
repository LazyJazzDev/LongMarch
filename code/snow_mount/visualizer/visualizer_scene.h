#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {

class Scene {
  Scene(const std::shared_ptr<Core> &core);

 public:
 private:
  std::shared_ptr<Core> core_;
};

}  // namespace snow_mount::visualizer
