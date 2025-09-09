#include "chang_zheng.h"

PYBIND11_MODULE(chang_zheng, m) {
  m.doc() = "ChangZheng library is targeting to be an all-in-one library for graphics study purpose.";

  m.attr("__version__") = "0.0.1";
  m.attr("BUILD_TIME") = __DATE__ " " __TIME__;

  pybind11::module_ m_cao_di = m.def_submodule("cao_di", "CaoDi");
  CD::PyBind(m_cao_di);

  pybind11::module_ m_xue_shan = m.def_submodule("xue_shan", "XueShan");
  XS::PyBind(m_xue_shan);
}
