#include "grassland/graphics/backend/d3d12/helper/root_signature.h"

namespace grassland::graphics::backend::d3d12 {
RootSignature::RootSignature(const ComPtr<ID3D12RootSignature> &root_signature) : root_signature_(root_signature) {
}
}  // namespace grassland::graphics::backend::d3d12
