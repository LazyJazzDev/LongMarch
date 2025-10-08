#include "long_march.h"

namespace long_march {

#if defined(LONGMARCH_PYTHON_ENABLED)
void PybindModuleRegistration(py::module_ &m) {
  m.doc() = "LongMarch library is designed for advanced graphics experiment.";
  m.def("hello", []() { py::print("Hello from LongMarch!"); });

  // submodul for graphics
  auto m_graphics = m.def_submodule("graphics", "RHI with Vulkan and D3D12 backends");
  graphics::PybindModuleRegistration(m_graphics);
}
#endif

}  // namespace long_march
