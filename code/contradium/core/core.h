#pragma once
#include "contradium/core/core_util.h"

namespace contradium {

class Core {
 public:
  Core(graphics::Core *core);

 private:
  graphics::Core *core_;
};

}  // namespace contradium
