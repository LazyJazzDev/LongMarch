#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12AllocatedBuffer, "f17438f1-4828-4dc2-943d-11e8afe77444")

namespace D3D12MA {
class Allocation;
class Allocator;
}  // namespace D3D12MA

namespace grassland::rhi::backend::d3d12 {
class ID3D12AllocatedBuffer : public ID3D12Buffer {
 public:
  static ID3D12AllocatedBuffer *Create(ID3D12Device *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size, uint64_t offset = 0) = 0;
};
}  // namespace grassland::rhi::backend::d3d12
