#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Core {
 public:
  Core(graphics::Core *core);

  sparkium::Core *GetRenderCore() const;

  contradium::Core *GetPhysicsCore() const;

 private:
  graphics::Core *core_;
  std::unique_ptr<sparkium::Core> render_core_;
  std::unique_ptr<contradium::Core> physics_core_;
};

}  // namespace practium
