#include "cao_di/util/util.h"

namespace CD {

void PyBindUtil(pybind11::module_ &m) {
  m.def("find_asset_file", &FindAssetFile);
  FPSCounter::PyBind(m);
}

}  // namespace CD
