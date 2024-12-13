#include "grassland/graphics/backend/d3d12/d3d12_program.h"

namespace grassland::graphics::backend {

D3D12Shader::D3D12Shader(const void *data, size_t size, D3D12Core *core)
    : core_(core) {
  D3DCreateBlob(size, &shader_blob_);
  memcpy(shader_blob_->GetBufferPointer(), data, size);
}

}  // namespace grassland::graphics::backend
