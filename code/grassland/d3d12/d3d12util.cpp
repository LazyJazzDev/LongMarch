#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {

std::string HRESULTToString(HRESULT hr) {
  char buffer[256] = {};
  sprintf(buffer, "HRESULT: 0x%08X", hr);
  return buffer;
}

}  // namespace grassland::d3d12
