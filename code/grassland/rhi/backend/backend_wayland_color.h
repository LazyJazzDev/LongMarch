#pragma once

#include <memory>

struct GLFWwindow;

namespace grassland::rhi::backend {

// Observes compositor feedback only; Vulkan WSI owns the surface description.
class WaylandColor {
 public:
  explicit WaylandColor(GLFWwindow *window);
  ~WaylandColor();
  bool ReferenceWhite(float &nits) const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace grassland::rhi::backend
