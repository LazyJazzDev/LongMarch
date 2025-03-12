#include "long_march.h"

PYBIND11_MODULE(long_march, m) {
  // set VERSION variable in python
  m.attr("__version__") = "0.0.1";
  m.attr("BUILD_TIME") = __DATE__ " " __TIME__;
}
