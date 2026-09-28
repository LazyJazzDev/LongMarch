#pragma once
#include "grassland/graphics/backend/d3d12/helper/d3d12util.h"

namespace grassland::graphics::backend {
class D3D12Core;
}

namespace grassland::graphics::backend::d3d12 {

class AccelerationStructure {
 public:
  AccelerationStructure(D3D12Core *core, const ComPtr<ID3D12Resource> &as, int num_instance);

  ID3D12Resource *Handle() const {
    return as_.Get();
  }

  HRESULT UpdateInstances(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> &instances,
                          ID3D12CommandQueue *queue,
                          ID3D12CommandAllocator *allocator);

  HRESULT UpdateInstances(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                          ID3D12CommandQueue *queue,
                          ID3D12CommandAllocator *allocator);

 private:
  D3D12Core *core_;
  ComPtr<ID3D12Resource> as_;
  int num_instance_;
};

}  // namespace grassland::graphics::backend::d3d12
