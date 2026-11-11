#include "grassland/rhi/rhi_window.h"

#include "grassland/rhi/backend/backend_glfw_helper.h"

namespace grassland::rhi {
IWindow::~IWindow() = default;

bool PollEvents() {
  return backend::glfw::PollEvents();
}
}  // namespace grassland::rhi

LONGMARCH_RHI_UUID(grassland::rhi::IWindow, "49b2b063-75db-4ba6-96ae-48d94b3a2022")
