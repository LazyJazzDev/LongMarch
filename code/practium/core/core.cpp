#include "practium/core/core.h"

namespace practium {

Core::Core(graphics::Core *core) : core_(core) {
  renderer_core_ = std::make_unique<sparkium::Core>(core_);
  physics_core_ = std::make_unique<contradium::Core>(core_);
}

}  // namespace practium
