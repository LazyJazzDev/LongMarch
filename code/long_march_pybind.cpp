#include "long_march.h"

PYBIND11_MODULE(long_march, m) {
  m.doc() = "LongMarch library is targeting to be an all-in-one library for graphics study purpose.";

  m.attr("__version__") = "0.0.1";
  m.attr("BUILD_TIME") = __DATE__ " " __TIME__;

  pybind11::module_ m_grassland = m.def_submodule("grassland", "Grassland");
  CD::PyBind(m_grassland);

  pybind11::module_ m_snow_mount = m.def_submodule("snow_mount", "SnowMount");
  XS::PyBind(m_snow_mount);
}
