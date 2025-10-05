#include <long_march.h>

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  practium::Core practium_core(core_.get());

  return 0;
}
