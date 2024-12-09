#include "app.h"

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{},
                                  &core_);

  int num_devices = core_->GetPhysicalDeviceProperties();
  std::vector<grassland::graphics::PhysicalDeviceProperties> properties(
      num_devices);

  core_->GetPhysicalDeviceProperties(properties.data());
  for (const auto &property : properties) {
    grassland::LogInfo("Device: {}, score: {}, ray tracing support: {}",
                       property.name, property.score,
                       property.ray_tracing_support);
  }
}

Application::~Application() {
  core_.reset();
}
