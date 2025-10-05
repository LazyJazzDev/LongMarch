#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Core {
 public:
  Core(graphics::Core *core);

 private:
  graphics::Core *core_;
  std::unique_ptr<sparkium::Core> renderer_core_;
  std::unique_ptr<contradium::Core> physics_core_;
};

}  // namespace practium
