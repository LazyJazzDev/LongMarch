#include "long_march.h"

PYBIND11_MODULE(long_march, m) {
  m.doc() = "LongMarch library is targeting to be an all-in-one library for graphics study purpose.";

  m.attr("__version__") = "0.0.1";
  m.attr("BUILD_TIME") = __DATE__ " " __TIME__;

  pybind11::module_ m_grassland = m.def_submodule("grassland", "Grassland");
  grassland::PyBind(m_grassland);

  pybind11::module_ m_snowberg = m.def_submodule("snowberg", "Snowberg");
  snowberg::PyBind(m_snowberg);
}
