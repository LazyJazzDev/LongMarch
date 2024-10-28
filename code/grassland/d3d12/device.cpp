#include "grassland/d3d12/device.h"

namespace grassland::d3d12 {

Device::Device(const class Adapter &adapter,
               const D3D_FEATURE_LEVEL feature_level,
               ID3D12Device *device)
    : adapter_(adapter), feature_level_(feature_level), device_(device) {
}

}  // namespace grassland::d3d12
