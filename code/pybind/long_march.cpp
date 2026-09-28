#include "pybind/grassland/graphics/graphics.h"

PYBIND11_MODULE(long_march, m) {
  m.doc() = "LongMarch library is designed for advanced graphics experiment.";
  m.def("hello", []() { py::print("Hello from LongMarch!"); });

  // submodul for graphics
  auto m_graphics = m.def_submodule("graphics", "RHI with Vulkan and D3D12 backends");
  grassland::graphics::pybind::RegisterGraphics(m_graphics);
}
