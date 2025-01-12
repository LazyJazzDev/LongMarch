#include "grassland/d3d12/raytracing/acceleration_structure.h"

namespace grassland::d3d12 {

AccelerationStructure::AccelerationStructure(const ComPtr<ID3D12Resource> &as) : as_(as) {
}

}  // namespace grassland::d3d12
