#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Core {
 public:
  static std::unique_ptr<Core> Create(rhi::IDevice *device);

  sparkium::Core *GetRenderCore() const;

  contradium::Core *GetPhysicsCore() const;

 private:
  Core() = default;
  std::unique_ptr<sparkium::Core> render_core_;
  std::unique_ptr<contradium::Core> physics_core_;
};

}  // namespace practium
