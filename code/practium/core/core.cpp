#include "practium/core/core.h"

namespace practium {

Core::Core(graphics::Core *core) : core_(core) {
  render_core_ = std::make_unique<sparkium::Core>(core_);
  physics_core_ = std::make_unique<contradium::Core>(core_);
}

sparkium::Core *Core::GetRenderCore() const {
  return render_core_.get();
}

contradium::Core *Core::GetPhysicsCore() const {
  return physics_core_.get();
}

}  // namespace practium
