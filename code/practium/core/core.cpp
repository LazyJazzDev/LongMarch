#include "practium/core/core.h"

namespace practium {

std::unique_ptr<Core> Core::Create(rhi::IDevice *device) {
  if (!device)
    return {};
  std::unique_ptr<Core> result(new Core);
  result->render_core_ = sparkium::Core::Create(device);
  if (!result->render_core_)
    return {};
  result->physics_core_ = std::make_unique<contradium::Core>(device);
  return result;
}

sparkium::Core *Core::GetRenderCore() const {
  return render_core_.get();
}

contradium::Core *Core::GetPhysicsCore() const {
  return physics_core_.get();
}

}  // namespace practium
