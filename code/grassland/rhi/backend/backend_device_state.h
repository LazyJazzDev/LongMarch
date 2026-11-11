#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct DeviceState final {
  DeviceState();
  ~DeviceState();
  std::array<ObjectPtr<IBackendQueue>, 2> queues_{};
  std::atomic<bool> stopping_{};
  BackendAPI api_{};
  IBackendCore *core_{};
  PhysicalDevice info_{};
  std::thread collector_{};
};
}  // namespace grassland::rhi::backend
