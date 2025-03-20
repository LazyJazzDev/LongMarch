#include "grassland/util/util.h"

namespace grassland {

void PyBindUtil(pybind11::module_ &m) {
  m.def("find_asset_file", &FindAssetFile);
}

}  // namespace grassland
