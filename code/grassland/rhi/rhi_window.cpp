#include "grassland/rhi/rhi_window.h"

#include "grassland/rhi/backend/backend_glfw_helper.h"

namespace grassland::rhi {
IWindow::~IWindow() = default;

bool PollEvents() {
  return backend::glfw::PollEvents();
}
}  // namespace grassland::rhi
