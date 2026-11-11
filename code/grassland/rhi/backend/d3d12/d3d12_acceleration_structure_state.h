#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"

namespace grassland::rhi::backend::d3d12 {
struct D3D12AccelerationStructureState final {
  D3D12AccelerationStructureState();
  ~D3D12AccelerationStructureState();
  ID3D12Device *device_{};
  ComPtr<ID3D12Resource> resource_{}, query_buffer_{}, query_output_{};
  ComPtr<D3D12MA::Allocation> allocation_{}, query_allocation_{}, query_output_allocation_{};
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs_{};
  uint64_t build_scratch_{}, update_scratch_{};
  bool allow_update_{}, allow_compaction_{}, compact_{};
};
}  // namespace grassland::rhi::backend::d3d12
