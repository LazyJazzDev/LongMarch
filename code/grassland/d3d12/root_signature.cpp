#include "grassland/d3d12/root_signature.h"

namespace CD::d3d12 {
RootSignature::RootSignature(const ComPtr<ID3D12RootSignature> &root_signature) : root_signature_(root_signature) {
}
}  // namespace CD::d3d12
