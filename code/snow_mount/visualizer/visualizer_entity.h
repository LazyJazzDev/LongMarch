#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {

class Entity : public std::enable_shared_from_this<Entity> {
  friend class Core;
};

}  // namespace snow_mount::visualizer
