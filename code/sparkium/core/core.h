#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {
class Core : public Object {
 public:
  Core(graphics::Core *core);

  graphics::Core *GraphicsCore() const;

 private:
  graphics::Core *core_{nullptr};
};
}  // namespace sparkium
